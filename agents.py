from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool
import logging
from dotenv import load_dotenv
import os

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# 导入工具
from tools.monitor_tools import (
    run_anomaly_detection,
    check_alert_duplicate,
    classify_alert_severity
)

ICE_API_KEY = os.getenv("ICE_API_KEY")
ICE_BASE_URL = os.getenv("ICE_BASE_URL")

gpt_llm = ChatOpenAI(
            model="gpt-5.6-luna",
            api_key=ICE_API_KEY,
            base_url=ICE_BASE_URL,
            temperature=0.0
        )
# MONITOR_AGENT的提示词
MONITOR_SYSTEM_PROMPT = """                                                                                                                                                                                                      
你是一个异常监测 Agent。你的职责是：                                                                                                                                                                                             

1. 检测时序数据中的异常                                                                                                                                                                                                          
   - 使用 run_anomaly_detection 工具分析指标历史数据                                                                                                                                                                             
   - 输出：算法投票结果、综合异常分数                                                                                                                                                                                            

2. 去重：检查告警是否重复                                                                                                                                                                                                        
   - 使用 check_alert_duplicate 工具                                                                                                                                                                                             
   - 同一个告警在 5 分钟内只算一次                                                                                                                                                                                               

3. 分级：对检测到的异常进行严重程度分级                                                                                                                                                                                          
   - 使用 classify_alert_severity 工具                                                                                                                                                                                           
   - 输出：CRITICAL / HIGH / MEDIUM / LOW                                                                                                                                                                                        

最后，你要返回一个结构化的告警结果，包含：                                                                                                                                                                                       
- is_anomaly: 是否检测到异常（bool）                                                                                                                                                                                             
- is_duplicate: 是否重复告警（bool）                                                                                                                                                                                             
- severity: 告警级别（str）                                                                                                                                                                                                      
- anomaly_score: 异常分数 0-10（float）                                                                                                                                                                                          
- service: 故障服务（str）                                                                                                                                                                                                       

开始工作吧！                                                                                                                                                                                                                     
"""

def create_monitor_agent(llm: ChatOpenAI = None):
    """
    创建 Monitor ReAct Agent。                                                                                                                                                                                                   

    Args:                                                                                                                                                                                                                        
        llm: LangChain LLM 实例（默认为 ChatOpenAI gpt-4o-mini）                                                                                                                                                                 

    Returns:                                                                                                                                                                                                                     
        一个 ReAct agent，可以调用 monitor_tools                                                                                                                                                                                 
    """
    # 初始化llm
    if llm is None:
        llm = gpt_llm
    # 收集工具列表
    tools = [run_anomaly_detection,check_alert_duplicate,classify_alert_severity]
    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt=MONITOR_SYSTEM_PROMPT
    )

    return agent

# RCA_Agent的提示词
RCA_SYSTEM_PROMPT = """                                                                                                                                                                                                          
  你是一个根因分析（Root Cause Analysis）Agent。你的职责是：                                                                                                                                                                       

  1. 查询服务依赖关系                                                                                                                                                                                                              
     - 使用 query_service_dependencies 查出某服务直接依赖哪些服务                                                                                                                                                                  

  2. 追踪故障影响链                                                                                                                                                                                                                
     - 使用 trace_impact_chain 看故障会波及哪些下游服务                                                                                                                                                                            

  3. 查找近期变更                                                                                                                                                                                                                  
     - 使用 find_recent_changes_in_service 看最近改过什么（最可能是根因）                                                                                                                                                          

  4. 列出根因候选                                                                                                                                                                                                                  
     - 使用 list_fault_candidates 根据告警类型列出可能的根因及其概率                                                                                                                                                               

  最后，综合这些信息，用贝叶斯推理计算后验概率，给出：                                                                                                                                                                             
  - root_cause: 最可能的根因                                                                                                                                                                                                       
  - confidence: 置信度（0-1）                                                                                                                                                                                                      
  - affected_services: 受影响的服务列表                                                                                                                                                                                            
  - suggested_actions: 建议的修复动作（如 restart_pod、rollback 等）                                                                                                                                                               

  不要返回 null，要分析到底！                                                                                                                                                                                                      
  """

def create_rca_agent(llm = None):
    """创建 RCA Agent"""
    # 初始化llm
    if llm is None:
        llm = gpt_llm

    # 导入四个工具放进工具列表
    from tools.rca_tools import (query_service_dependencies,trace_impact_chain,find_recent_changes_in_service,list_fault_candidates)

    tools = [query_service_dependencies, trace_impact_chain, find_recent_changes_in_service, list_fault_candidates]

    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(model=llm,tools=tools,system_prompt=RCA_SYSTEM_PROMPT)

    return agent

# HEAL_Agent的提示词
HEAL_SYSTEM_PROMPT = """                                                                                                                                                                                                         
  你是一个自愈执行 Agent。你的职责是：                                                                                                                                                                                             

  1. 检查熔断器状态                                                                                                                                                                                                                
     - 使用 check_circuit_breaker_status 确保系统不在故障模式中                                                                                                                                                                    
     - 如果熔断器打开（OPEN），暂停自愈，等待恢复                                                                                                                                                                                  

  2. 匹配修复方案                                                                                                                                                                                                                  
     - 使用 match_remediation_playbook 从建议的修复动作中选出最合适的（优先低风险）                                                                                                                                                

  3. 模拟执行（干运行）                                                                                                                                                                                                            
     - 使用 simulate_dry_run 模拟执行修复命令，不实际修改系统                                                                                                                                                                      
     - 验证命令的正确性和可行性                                                                                                                                                                                                    

  4. 记录执行结果                                                                                                                                                                                                                  
     - 使用 record_heal_result 记录修复是否成功                                                                                                                                                                                    
     - 更新熔断器状态                                                                                                                                                                                                              

  最后返回：                                                                                                                                                                                                                       
  - action: 执行的修复动作                                                                                                                                                                                                         
  - status: SUCCESS / FAILED / PENDING_APPROVAL                                                                                                                                                                                    
  - dry_run_output: 模拟执行的输出                                                                                                                                                                                                 
  - circuit_breaker_state: 熔断器状态                                                                                                                                                                                              

  记住：总是先做干运行，再决定是否真实执行！                                                                                                                                                                                       
  """

def create_heal_agent(llm = None):
    if llm is None:
        llm = gpt_llm

    from tools.heal_tools import (check_circuit_breaker_status,match_remediation_playbook,simulate_dry_run,record_heal_result)

    tools = [check_circuit_breaker_status,match_remediation_playbook,simulate_dry_run,record_heal_result]

    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(model=llm,tools=tools,system_prompt=HEAL_SYSTEM_PROMPT)

    return agent