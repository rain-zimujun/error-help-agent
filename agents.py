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
ICE_API_KEY = os.getenv("ICE_API_KEY")
ICE_BASE_URL = os.getenv("ICE_BASE_URL")

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
        llm = ChatOpenAI(
            model="gpt-5.6-luna",
            api_key=ICE_API_KEY,
            base_url=ICE_BASE_URL,
            temperature=0.0
        )
    # 收集工具列表
    tools = [run_anomaly_detection,check_alert_duplicate,classify_alert_severity]
    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(
        llm_with_tools,
        tools,
        system_prompt=MONITOR_SYSTEM_PROMPT
    )

    return agent