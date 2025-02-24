#include "GlobalPPscheduler.hh"
#include "Workload.hh"

namespace AstraSim
{

    //
    // Actionphase（）分为四种模式
    // Actionphase(Mode, nodeid, Workload::LoopState::)
    // 0模式：不通信
    // 1模式：单向通信，nodeid为目标节点
    // 2模式：双向通信，向nodeid发送，从nodeid接收
    // 3模式: 睡眠，等待到达后计算
    // 4模式: 单向通信, continue forward pass

    // Warm-up逻辑：
    // 在Warm-up阶段，我们还没有进入1F1B模式。此时逻辑是：
    // - 如果当前节点（stage）已经完成了(m_pp_size - 1)次forward传递，那么说明流水线已填满，
    //   切换到1F1B阶段。
    // - 否则，继续发起Forward Pass，传给下一个节点。
    Actionphase GlobalPPscheduler::warm_up(int nodeid, int state, int pp_index, std::vector<int> ranks)
    {
        std::ofstream outFile("output.txt", std::ios::app);
        current_state_per_node_map[nodeid]->current_forward_batch_number++;
        int current_forward_minibatch = current_state_per_node_map[nodeid]->current_forward_batch_number;
        int pipeline_step = current_state_per_node_map[nodeid]->pipeline_step;
        // bool forward_reach = current_state_per_node_map[nodeid].forward_reach;
        // bool backward_reach = current_state_per_node_map[nodeid].backward_reach;
        int next_nodeid = nodeid + m_pp_num;
        // int prev_nodeid = nodeid - m_pp_num;

        if (current_forward_minibatch > m_pp_size - pipeline_step - 1)
        {
            current_state_per_node_map[nodeid]->pipeline_stage = PipelineStage::_1F1B;
            if (current_forward_minibatch == m_mini_batch)
            {
                outFile << "1F1B Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(3, nodeid, int(Workload::LoopState::Input_Gradient));
            }

            if (pipeline_step == m_pp_size - 1)
            {
                // current_state_per_node_map[nodeid]->current_backward_batch_number++;
                // 此时需要最后一阶段跳过通信过程直接开启后向计算，后向计算是否为weight_gradient
                outFile << "1F1B Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(0, nodeid, int(Workload::LoopState::Input_Gradient));
            }
            else
            {
                // current_state_per_node_map[nodeid]->current_backward_batch_number++;
                // 需要等待下游的后向计算到达后在开启本节点的后向计算
                outFile << "1F1B Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(3, nodeid, int(Workload::LoopState::Input_Gradient));
            }
        }
        else
        {
            if (pipeline_step == 0)
            {
                Actionphase a = Actionphase(4, next_nodeid, int(Workload::LoopState::Forward_Pass));
                if (a.m_next_commu_node_id < 0 || a.m_next_commu_node_id >= 1024)
                {
                    printf("warm_up: actionphase.m_next_commu_node_id == -1\n");
                }
                outFile << "Warm up working Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Forward" << std::endl;
                return a;
            }
            else
            {
                Actionphase a = Actionphase(1, next_nodeid, int(Workload::LoopState::Forward_Pass));
                if (a.m_next_commu_node_id < 0 || a.m_next_commu_node_id >= 1024)
                {
                    printf("warm_up: actionphase.m_next_commu_node_id == -1\n");
                }
                outFile << "Warm up working Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Forward" << std::endl;
                return a;
            }
        }
    }

    // 1F1B阶段逻辑：
    // 在1F1B阶段，每一步都期望有1个forward和1个backward进行。
    // 具体规则：
    // - 如果还有未完成的forward（forward_count < m_mini_batch），
    //   本节点继续发起下一批forward。
    // - 同时，如果本节点有可执行的backward（backward_count < forward_count），
    //   则可以进行backward。
    // 当forward全部发起完 (forward_count == m_mini_batch)，进入cool_down阶段。
    // 在1F1B过程中：
    // - 如果本节点是第一个stage，它会负责发起forward pass的micro-batch。
    // - 如果本节点是最后一个stage，它在1F1B阶段也会频繁完成forward_reach并发起backward_reach。
    Actionphase GlobalPPscheduler::A_1F1B(int nodeid, int state, int pp_index)
    {
        std::ofstream outFile("output.txt", std::ios::app);
        int current_forward_minibatch = current_state_per_node_map[nodeid]->current_forward_batch_number;
        int current_backward_minibatch = current_state_per_node_map[nodeid]->current_backward_batch_number;
        int pipeline_step = current_state_per_node_map[nodeid]->pipeline_step;
        // bool forward_reach = current_state_per_node_map[nodeid]->forward_reach;
        // bool forward_go = current_state_per_node_map[nodeid]->forward_go;
        // bool backward_reach = current_state_per_node_map[nodeid]->backward_reach;
        int next_nodeid = nodeid + m_pp_num;
        int prev_nodeid = nodeid - m_pp_num;

        if (current_forward_minibatch >= m_mini_batch)
        {
            if (pipeline_step == m_pp_size - 1)
            {
                current_state_per_node_map[nodeid]->current_backward_batch_number++;
                current_state_per_node_map[nodeid]->pipeline_stage = PipelineStage::Cool_Down;
                // 不需要通信，开启后向计算
                outFile << "Cool Down Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(1, prev_nodeid, int(Workload::LoopState::Input_Gradient));
            }

            if (pipeline_step == 0)
            {
                current_state_per_node_map[nodeid]->current_backward_batch_number++;
                current_state_per_node_map[nodeid]->pipeline_stage = PipelineStage::Cool_Down;
                // 不需要通信，开启后向计算
                outFile << "Cool Down Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(3, nodeid, int(Workload::LoopState::Input_Gradient));
            }

            else
            {
                current_state_per_node_map[nodeid]->current_backward_batch_number++;
                current_state_per_node_map[nodeid]->pipeline_stage = PipelineStage::Cool_Down;
                outFile << "Cool Down Begin Current Node id: " << nodeid
                        << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                        << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                        << ", Next Loopstate: Backward" << std::endl;
                return Actionphase(1, prev_nodeid, int(Workload::LoopState::Input_Gradient));
            }
        }
        else
        {
            if (pipeline_step == m_pp_size - 1)
            {
                if (current_backward_minibatch == 0)
                {
                    current_state_per_node_map[nodeid]->current_backward_batch_number++;
                    // 同时开启对上游后向的发送和对上游前向的接收
                    outFile << "1F1B working Current Node id: " << nodeid
                            << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                            << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                            << ", Next Loopstate: Forward" << std::endl;
                    return Actionphase(2, prev_nodeid, int(Workload::LoopState::Forward_Pass));
                }
                else
                {
                    if (current_backward_minibatch == current_forward_minibatch)
                    {
                        current_state_per_node_map[nodeid]->current_forward_batch_number++;
                        // 同时开启对上游后向的发送和对上游前向的接收
                        outFile << "1F1B working Current Node id: " << nodeid
                                << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                                << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                                << ", Next Loopstate: Forward" << std::endl;
                        return Actionphase(0, nodeid, int(Workload::LoopState::Input_Gradient));
                    }
                    else
                    {
                        current_state_per_node_map[nodeid]->current_backward_batch_number++;
                        outFile << "1F1B working Current Node id: " << nodeid
                                << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                                << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                                << ", Next Loopstate: Backward" << std::endl;
                        return Actionphase(1, prev_nodeid, int(Workload::LoopState::Forward_Pass));
                    }
                }
            }
            else if (pipeline_step == 0)
            {
                if (current_forward_minibatch - current_backward_minibatch < m_pp_size)
                {
                    current_state_per_node_map[nodeid]->current_forward_batch_number++;
                    outFile << "1F1B working Current Node id: " << nodeid
                            << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                            << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                            << ", Next Loopstate: Forward" << std::endl;
                    return Actionphase(1, next_nodeid, int(Workload::LoopState::Input_Gradient));
                }
                else
                {
                    current_state_per_node_map[nodeid]->current_backward_batch_number++;
                    // 同时开启对下游前向的发送和对下游后向的接收
                    outFile << "1F1B working Current Node id: " << nodeid
                            << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                            << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                            << ", Next Loopstate: Backward" << std::endl;           
                    return Actionphase(0, nodeid, int(Workload::LoopState::Forward_Pass));
                }
            }
            else
            {
                if (current_backward_minibatch == 0)
                {
                    current_state_per_node_map[nodeid]->current_backward_batch_number++;
                    outFile << "1F1B working Current Node id: " << nodeid
                            << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                            << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                            << ", Next Loopstate: Forward" << std::endl;
                    return Actionphase(2, prev_nodeid, int(Workload::LoopState::Forward_Pass));
                }
                else
                {
                    if (current_forward_minibatch + pipeline_step - current_backward_minibatch < m_pp_size)
                    {
                        current_state_per_node_map[nodeid]->current_forward_batch_number++;
                        outFile << "1F1B working Current Node id: " << nodeid
                                << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                                << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                                << ", Next Loopstate: Forward" << std::endl;
                        return Actionphase(1, next_nodeid, int(Workload::LoopState::Input_Gradient));
                    }
                    else
                    {
                        current_state_per_node_map[nodeid]->current_backward_batch_number++;
                        outFile << "1F1B working Current Node id: " << nodeid
                                << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                                << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                                << ", Next Loopstate: Backward" << std::endl;
                        return Actionphase(1, prev_nodeid, int(Workload::LoopState::Forward_Pass));
                    }
                }
            }
        }
    }
    // cool_down逻辑：
    // 当所有forward已经发完，只剩下pipeline中尚未完成的backward需要清空。
    // 在cool_down阶段：
    // - 没有新的forward发起
    // - 只进行backward传递，直到backward_count == m_mini_batch
    Actionphase GlobalPPscheduler::cool_down(int nodeid, int state, int pp_index)
    {
        std::ofstream outFile("output.txt", std::ios::app);
        int current_backward_minibatch = current_state_per_node_map[nodeid]->current_backward_batch_number;
        int pipeline_step = current_state_per_node_map[nodeid]->pipeline_step;
        int prev_nodeid = nodeid - m_pp_num;
        bool backward_reach = current_state_per_node_map[nodeid]->backward_reach;
        // if (current_backward_minibatch >= m_mini_batch)
        // {
        //     if (pipeline_step == 0)
        //     {
        //         // printf("Data Parallel Communication Begin !!!!!!\n");
        //         outFile << "final Cool Down working Current Node id: " << nodeid
        //         << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
        //         << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
        //         << ", Next Loopstate: Backward" << std::endl;
        //         return Actionphase(3,nodeid, int (Workload::LoopState::Input_Gradient));
        //     }
        //     else
        //     {
        //         // printf("Data Parallel Communication Begin !!!!!!\n");
        //         outFile << "Cool Down working Current Node id: " << nodeid
        //                 << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
        //                 << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
        //                 << ", Next Loopstate: Backward" << std::endl;
        //         return Actionphase(1, prev_nodeid, int(Workload::LoopState::Input_Gradient));
        //     }
        // }
        // else
        // {
        if (pipeline_step == 0)
        {
            current_state_per_node_map[nodeid]->current_backward_batch_number++;
            outFile << "Cool Down working Current Node id: " << nodeid
                    << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                    << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                    << ", Next Loopstate: Backward" << std::endl;
            return Actionphase(3, nodeid, int(Workload::LoopState::Input_Gradient));
        }
        else
        {
            current_state_per_node_map[nodeid]->current_backward_batch_number++;
            outFile << "Cool Down working Current Node id: " << nodeid
                    << ", Current forward batch number: " << current_state_per_node_map[nodeid]->current_forward_batch_number
                    << ", Current backward batch number: " << current_state_per_node_map[nodeid]->current_backward_batch_number
                    << ", Next Loopstate: Backward" << std::endl;
            return Actionphase(1, prev_nodeid, int(Workload::LoopState::Input_Gradient));
        }
        // }
    }

    void GlobalPPscheduler::finalize()
    {
        // Finalization code
    }

    Actionphase GlobalPPscheduler::getactionsby_nodeid(int nodeid, int state, std::vector<int> ranks)
    {
        // 加锁保护共享数据
        {
            std::lock_guard<std::mutex> lock(mtx);

            // 检查并插入新节点
            if (current_state_per_node_map.find(nodeid) == current_state_per_node_map.end())
            {
                int pp_step = nodeid / m_pp_num;
                current_state_per_node_map[nodeid] = new CurrentState(
                    0, int(Workload::LoopState::Forward_Pass),
                    pp_step, PipelineStage::Warm_UP);
            }
        }

        // 根据状态执行操作，需再次加锁读取
        Actionphase result;
        {
            std::lock_guard<std::mutex> lock(mtx);

            auto currentState = current_state_per_node_map[nodeid];
            switch (static_cast<int>(currentState->pipeline_stage))
            {
            case int(PipelineStage::Warm_UP):
                result = warm_up(nodeid, state, currentState->pipeline_step, ranks);
                if (result.m_next_commu_node_id < 0 || result.m_next_commu_node_id >= 1024)
                {
                    printf("warm_up: actionphase.m_next_commu_node_id == -1\n");
                }
                break;

            case int(PipelineStage::_1F1B):
                result = A_1F1B(nodeid, state, currentState->pipeline_step);
                break;

            case int(PipelineStage::Cool_Down):
                result = cool_down(nodeid, state, currentState->pipeline_step);
                break;

            default:
                result = Actionphase(0, nodeid, state);
                break;
            }
        }

        std::ofstream outFile("output1.txt", std::ios::app);
        outFile << "Nodeid: " << nodeid
                << ", mode: " << result.m_mode
                << ", c_id " << result.m_next_commu_node_id
                << ", next_state " << result.m_next_state
                << ", forward " << current_state_per_node_map[nodeid]->current_forward_batch_number
                << ", backward " << current_state_per_node_map[nodeid]->current_backward_batch_number
                << ", time " << Sys::boostedTick()
                << std::endl;

        return result;
    }

    // int GlobalPPscheduler::get_pipeline_step(int nodeid)
    // {
    //     return 0;
    // }
}
// namespace astra_sim