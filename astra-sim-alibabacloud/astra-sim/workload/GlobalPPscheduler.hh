#ifndef __globalpp_HH__
#define __globalpp_HH__

// #include "Workload.hh"
// #include "CSVWriter.hh"
// #include "Layer.hh"

#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include "Workload.hh"

namespace AstraSim
{

    class Workload;

    enum class PipelineStage
    {
        Warm_UP,
        _1F1B,
        Cool_Down
    };

    class Actionphase
    {
    public:
        int m_mode;
        int m_next_commu_node_id;
        int m_next_state;
        Actionphase()= default;
        Actionphase(int mode, int next_commu_node_id, int next_state) :m_mode(mode),m_next_commu_node_id(next_commu_node_id), m_next_state(next_state) {};
        ~Actionphase()= default;;
    };

    class CurrentState
    {
    public:
    //每个node的状态变量
        int current_forward_batch_number;//当前已处理的forward micro-batch数
        int current_backward_batch_number=0;//当前已处理的backward micro-batch数
        int m_current_state;
        int pipeline_step;//当前节点在流水线中间的位置,从0开始
        PipelineStage pipeline_stage;//当前所处阶段（WarmUp, _1F1B, CoolDown）
        bool forward_go ;//阶段一是否能发出新的minibatch
        bool forward_reach;//当前node是否已收到上游的forward结果,用于决定是否可开始forward
        bool backward_reach;//当前node是否已收到下游的backward结果,用于决定是否可开始backward

        CurrentState(int mini_batch_number, int current_state, int m_pipeline_step, PipelineStage m_pipeline_stage)
            : current_forward_batch_number(mini_batch_number), m_current_state(current_state),pipeline_step(m_pipeline_step),pipeline_stage(m_pipeline_stage) {};

            // Default constructor
         CurrentState() : current_forward_batch_number(0), m_current_state(0), pipeline_step(0) {};

        ~CurrentState()= default;;
    };



    class GlobalPPscheduler
    {
    public:
        int m_pp_size;
        int m_pp_num;
        int m_mini_batch;
        std::mutex mtx;  // 定义互斥锁


        std::map<int, PipelineStage> pipeline_stage_map;

        std::map<int, CurrentState*> current_state_per_node_map;

        Actionphase getactionsby_nodeid(int modeid, int type,std::vector<int>);
        GlobalPPscheduler()= default;;
        GlobalPPscheduler(int pp_size, int pp_num, int mini_batch): m_pp_size(pp_size), m_pp_num(pp_num), m_mini_batch(mini_batch) {};
        ~GlobalPPscheduler()= default;;
        void initialize();
        void schedule();
        Actionphase warm_up(int nodeid, int type, int pipeline_step,std::vector<int>);
        Actionphase A_1F1B(int nodeid, int state, int pp_index);
        Actionphase cool_down(int nodeid, int state, int pp_index);
        void pipeline_schedule(int mini_batch, int pp_size);
        void cool_down(int cool_down_steps);
        void finalize();
        void update_batch_numbers(int nodeid, bool is_forward);

        static GlobalPPscheduler &getInstance()
        {
            static GlobalPPscheduler instance;
            return instance;
        }
    };
}

#endif  