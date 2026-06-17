/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <limits>
#include <string>

#include "acl/acl_rt.h"
#include "collective_kernel.h"
#include "collective_launcher.h"
#include "collective_utils.h"
#include "tilexr_collectives.h"

namespace {

constexpr int kReduceScatterLocalTreeAlgo = 1;

int ValidateCommon(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm)
{
    if (comm == nullptr || sendBuf == nullptr || recvBuf == nullptr || sendCount <= 0 ||
        !TileXRCollectives::Host::IsSupportedDataType(dataType)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (TileXRCollectives::Host::CountToBytes(sendCount, dataType) < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int ValidateReduce(void *sendBuf, void *recvBuf, int64_t count,
                   TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                   TileXRCommPtr comm)
{
    const int ret = ValidateCommon(sendBuf, recvBuf, count, dataType, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXRCollectives::Host::IsSupportedReduceOp(op) ?
        TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
}

int ValidateBroadcastLocal(void *buf, int64_t count, TileXR::TileXRDataType dataType,
                           int root, TileXRCommPtr comm)
{
    if (comm == nullptr || buf == nullptr || count <= 0 ||
        !TileXRCollectives::Host::IsSupportedDataType(dataType) || root < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (TileXRCollectives::Host::CountToBytes(count, dataType) < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int LoopbackCopy(void *sendBuf, void *recvBuf, int64_t bytes, aclrtStream stream)
{
    if (sendBuf == recvBuf) {
        return TileXR::TILEXR_SUCCESS;
    }
    const aclError ret = aclrtMemcpyAsync(recvBuf, static_cast<size_t>(bytes), sendBuf, static_cast<size_t>(bytes),
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    return ret == ACL_SUCCESS ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_INTERNAL;
}

bool EnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

bool CanUseReduceScatterLocalTree(const TileXR::CommArgs &args, int64_t bytesPerRank)
{
    if (!EnvEnabled("TILEXR_REDUCE_SCATTER_LOCAL_TREE")) {
        return false;
    }
    if (args.rankSize <= 1 || bytesPerRank <= 0) {
        return false;
    }
    if ((args.extraFlag & (TileXR::ExtraFlag::TOPO_PCIE | TileXR::ExtraFlag::TOPO_910_93)) != 0) {
        return false;
    }
    const int64_t workspaceFactor = 2 * static_cast<int64_t>(args.rankSize);
    return workspaceFactor > 0 && bytesPerRank <= TileXR::IPC_BUFF_MAX_SIZE / workspaceFactor;
}

} // namespace

int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream)
{
    int ret = ValidateCommon(sendBuf, recvBuf, sendCount, dataType, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(sendCount, dataType);
    if (context.hostArgs->rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, bytes, stream);
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetAllGatherBlockNum(*context.hostArgs, bytes);
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::ALL_GATHER, context,
        sendBuf, recvBuf, sendCount, dataType, blockDim, stream);
}

int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                   aclrtStream stream)
{
    int ret = ValidateCommon(sendBuf, recvBuf, sendCount, dataType, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t loopbackBytes = TileXRCollectives::Host::CountToBytes(sendCount, dataType);
    const int rankSize = context.hostArgs->rankSize;
    if (rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, loopbackBytes, stream);
    }
    if (rankSize % TileXR::RANK_SIZE_TWO != 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (sendCount > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(rankSize)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t kernelCount = sendCount * static_cast<int64_t>(rankSize);
    const int64_t kernelBytes = TileXRCollectives::Host::CountToBytes(kernelCount, dataType);
    if (kernelBytes < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const uint32_t blockDim = TileXRCollectives::Host::GetAllToAllBlockNum(*context.hostArgs, kernelBytes);
    if (blockDim == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::ALL2ALL, context,
        sendBuf, recvBuf, kernelCount, dataType, blockDim, stream);
}

int TileXRAllReduce(void *sendBuf, void *recvBuf, int64_t count,
                    TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                    TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateReduce(sendBuf, recvBuf, count, dataType, op, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(count, dataType);
    if (context.hostArgs->rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, bytes, stream);
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetAllReduceBlockNum(*context.hostArgs, bytes);
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::ALL_REDUCE, context,
        sendBuf, recvBuf, count, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { static_cast<int>(op), 0 });
}

int TileXRReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount,
                        TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                        TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateReduce(sendBuf, recvBuf, recvCount, dataType, op, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(recvCount, dataType);
    const int rankSize = context.hostArgs->rankSize;
    if (rankSize <= 1) {
        return LoopbackCopy(sendBuf, recvBuf, bytes, stream);
    }
    if (recvCount > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(rankSize)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t inputCount = recvCount * static_cast<int64_t>(rankSize);
    const int64_t inputBytes = TileXRCollectives::Host::CountToBytes(inputCount, dataType);
    if (inputBytes < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const bool useLocalTree = CanUseReduceScatterLocalTree(*context.hostArgs, bytes);
    const uint32_t blockDim = useLocalTree ? static_cast<uint32_t>(rankSize) :
        TileXRCollectives::Host::GetReduceScatterBlockNum(*context.hostArgs, bytes);
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::REDUCE_SCATTER, context,
        sendBuf, recvBuf, recvCount, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { static_cast<int>(op),
            useLocalTree ? kReduceScatterLocalTreeAlgo : 0 });
}

int TileXRBroadcast(void *buf, int64_t count,
                    TileXR::TileXRDataType dataType, int root,
                    TileXRCommPtr comm, aclrtStream stream)
{
    int ret = ValidateBroadcastLocal(buf, count, dataType, root, comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    ret = TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (root >= context.hostArgs->rankSize) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t bytes = TileXRCollectives::Host::CountToBytes(count, dataType);
    if (context.hostArgs->rankSize <= 1) {
        return TileXR::TILEXR_SUCCESS;
    }

    const uint32_t blockDim = TileXRCollectives::Host::GetBroadcastBlockNum(*context.hostArgs, bytes);
    if (blockDim == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXRCollectives::Host::LaunchCollectiveKernel(comm, TileXR::TileXRType::BROADCAST, context,
        buf, buf, bytes, dataType, blockDim, stream,
        TileXRCollectives::Host::CollectiveLaunchAttrs { 0, root });
}
