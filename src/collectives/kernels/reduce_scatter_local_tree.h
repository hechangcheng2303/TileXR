/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILEXR_REDUCE_SCATTER_LOCAL_TREE_H
#define TILEXR_REDUCE_SCATTER_LOCAL_TREE_H

#include "collectives.h"

using namespace AscendC;

constexpr int TILEXR_REDUCE_SCATTER_ALGO_LOCAL_TREE = 1;

template<typename T>
class ReduceScatterLocalTree : protected Collectives {
public:
    FORCE_INLINE_AICORE ReduceScatterLocalTree(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        atomOp = op;
        rankSizeU32 = static_cast<uint32_t>(rankSize);
        lenPerRank = len;
        valid = rankSizeU32 > 1 && blockNum >= rankSizeU32 && atomOp == ADD;
        if (!valid) {
            return;
        }

        const int64_t bytesPerRank = lenPerRank * static_cast<int64_t>(sizeof(T));
        valid = bytesPerRank > 0 && rankSizeU32 <= TILEXR_MAX_RANK_SIZE &&
            lenPerRank <= static_cast<int64_t>(0xFFFFFFFFULL) &&
            bytesPerRank <= IPC_BUFF_MAX_SIZE / (2 * static_cast<int64_t>(rankSizeU32));
    }

    FORCE_INLINE_AICORE void Process()
    {
        if (!valid) {
            return;
        }

        PublishLocalShard();
        FetchPeerShardToLocalStage();
        SyncAll<true>();
        LocalTreeReduce();
        StoreResult();
        WaitAllFetchDone();
    }

private:
    constexpr static int32_t STAGE_READY_EVENT_BASE = TILEXR_MAX_RANK_SIZE;
    constexpr static int32_t REDUCE_READY_EVENT_BASE = 2 * TILEXR_MAX_RANK_SIZE;
    constexpr static int32_t PEER_FETCH_DONE_EVENT_BASE = 3 * TILEXR_MAX_RANK_SIZE;

    bool valid = false;
    int atomOp = COPYONLY;
    int64_t lenPerRank = 0;
    uint32_t rankSizeU32 = 0;

    FORCE_INLINE_AICORE int64_t LocalPublishOffset(uint32_t targetRank) const
    {
        return static_cast<int64_t>(targetRank) * lenPerRank;
    }

    FORCE_INLINE_AICORE int64_t LocalStageOffset(uint32_t peerRank) const
    {
        return static_cast<int64_t>(rankSizeU32 + peerRank) * lenPerRank;
    }

    FORCE_INLINE_AICORE uint32_t CeilLog2(uint32_t n) const
    {
        if (n <= 1) {
            return 0;
        }
        uint32_t result = 0;
        uint32_t value = 1;
        while (value < n) {
            value <<= 1;
            ++result;
        }
        return result;
    }

    FORCE_INLINE_AICORE uint32_t GetLargestPowerOf2(uint32_t n) const
    {
        if (n <= 1) {
            return 0;
        }
        uint32_t result = 1;
        while ((result << 1) < n) {
            result <<= 1;
        }
        return result;
    }

    FORCE_INLINE_AICORE void PublishLocalShard()
    {
        const uint32_t targetRank = static_cast<uint32_t>(blockIdx);
        if (targetRank >= rankSizeU32) {
            return;
        }
        GlobalTensor<T> src;
        GlobalTensor<T> dst;
        src.SetGlobalBuffer((__gm__ T*)input + targetRank * lenPerRank, lenPerRank);
        dst.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + LocalPublishOffset(targetRank),
            lenPerRank);
        CpGM2GM<T>(dst, src, static_cast<uint32_t>(lenPerRank), COPYONLY);
        sync.SetSyncFlag(static_cast<int32_t>(magic), rank, targetRank, rank);
    }

    FORCE_INLINE_AICORE void FetchPeerShardToLocalStage()
    {
        const uint32_t peerRank = static_cast<uint32_t>(blockIdx);
        if (peerRank >= rankSizeU32) {
            return;
        }
        sync.WaitSyncFlag(static_cast<int32_t>(magic), peerRank, rank, peerRank);

        GlobalTensor<T> src;
        GlobalTensor<T> dst;
        src.SetGlobalBuffer((__gm__ T*)(shareAddrs[peerRank] + IPC_DATA_OFFSET) + LocalPublishOffset(rank),
            lenPerRank);
        dst.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + LocalStageOffset(peerRank),
            lenPerRank);
        CpGM2GM<T>(dst, src, static_cast<uint32_t>(lenPerRank), COPYONLY);
        sync.SetSyncFlag(static_cast<int32_t>(magic), rank, STAGE_READY_EVENT_BASE + peerRank, rank);
        sync.SetSyncFlag(static_cast<int32_t>(magic), rank, PEER_FETCH_DONE_EVENT_BASE + rank, peerRank);
    }

    FORCE_INLINE_AICORE void WaitAllFetchDone()
    {
        const uint32_t peerRank = static_cast<uint32_t>(blockIdx);
        if (peerRank >= rankSizeU32) {
            return;
        }
        sync.WaitSyncFlag(static_cast<int32_t>(magic), peerRank, PEER_FETCH_DONE_EVENT_BASE + peerRank, rank);
    }

    FORCE_INLINE_AICORE void LocalTreeReduce()
    {
        uint32_t curBlocks = rankSizeU32;
        const uint32_t totalRounds = CeilLog2(rankSizeU32);
        for (uint32_t round = 0; round < totalRounds; ++round) {
            const uint32_t powerOf2 = GetLargestPowerOf2(curBlocks);
            if (powerOf2 == 0) {
                break;
            }
            const uint32_t offset = static_cast<uint32_t>(blockIdx);
            if (offset < powerOf2) {
                const uint32_t backIdx = powerOf2 + offset;
                if (backIdx < curBlocks) {
                    if (round == 0) {
                        sync.WaitSyncFlag(static_cast<int32_t>(magic), rank, STAGE_READY_EVENT_BASE + offset, rank);
                        sync.WaitSyncFlag(static_cast<int32_t>(magic), rank, STAGE_READY_EVENT_BASE + backIdx, rank);
                    } else {
                        sync.WaitSyncFlag(static_cast<int32_t>(magic), round, REDUCE_READY_EVENT_BASE + offset, rank);
                        sync.WaitSyncFlag(static_cast<int32_t>(magic), round, REDUCE_READY_EVENT_BASE + backIdx, rank);
                    }

                    GlobalTensor<T> front;
                    GlobalTensor<T> back;
                    front.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + LocalStageOffset(offset),
                        lenPerRank);
                    back.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + LocalStageOffset(backIdx),
                        lenPerRank);
                    CpGM2GM<T>(front, back, static_cast<uint32_t>(lenPerRank), atomOp);
                }
                sync.SetSyncFlag(static_cast<int32_t>(magic), static_cast<int32_t>(round + 1),
                    REDUCE_READY_EVENT_BASE + offset, rank);
            }
            curBlocks = powerOf2;
            SyncAll<true>();
        }
    }

    FORCE_INLINE_AICORE void StoreResult()
    {
        if (blockIdx != 0) {
            return;
        }
        GlobalTensor<T> src;
        GlobalTensor<T> dst;
        src.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + LocalStageOffset(0), lenPerRank);
        dst.SetGlobalBuffer((__gm__ T*)output, lenPerRank);
        CpGM2GM<T>(dst, src, static_cast<uint32_t>(lenPerRank), COPYONLY);
    }
};

#endif // TILEXR_REDUCE_SCATTER_LOCAL_TREE_H
