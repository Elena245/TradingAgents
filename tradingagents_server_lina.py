"""
TradingAgents Backend API Server
基于 lina_test.py 的 Flask 服务器，提供前端 API 接口

API 端点：
1. POST /api/analyze - 启动新的分析
   请求体：
   {
     "ticker": "SPY",
     "analysis_date": "2024-01-01",
     "analysts": ["market", "social"],
     "research_depth": 2,
     "llm_provider": "openai",
     "backend_url": null,
     "shallow_thinker": "gpt-4o-mini",
     "deep_thinker": "gpt-4o-mini",
     "translateContent": true
   }
   返回：
   {
     "success": true,
     "analysisId": "uuid",
     "message": "Analysis started"
   }

2. GET /api/analysis/<analysis_id>/status - 获取分析状态（实时更新）
   返回：
   {
     "analysisId": "uuid",
     "status": "running|completed|failed",
     "progress": 0-100,
     "currentStage": "当前阶段",
     "error": null,
     "displayData": {...}  // 实时显示数据
   }

3. GET /api/analysis/<analysis_id>/results - 获取最终结果
   返回：
   {
     "analysisId": "uuid",
     "result": {...}  // 完整的显示数据
   }

4. GET /api/analysis - 列出所有分析

5. POST /api/analysis/<analysis_id>/cancel - 取消正在进行的分析
   返回：
   {
     "success": true,
     "analysisId": "uuid",
     "message": "Analysis cancelled successfully",
     "status": "cancelled"
   }

6. GET /api/health - 健康检查

使用示例：
  # 启动分析
  curl -X POST http://localhost:5000/api/analyze \
    -H "Content-Type: application/json" \
    -d '{"ticker": "SPY", "analysis_date": "2024-01-01", "analysts": ["market"]}'
  
  # 获取状态
  curl http://localhost:5000/api/analysis/<analysis_id>/status
  
  # 获取结果
  curl http://localhost:5000/api/analysis/<analysis_id>/results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from datetime import datetime
import uuid
import traceback
import os
from pathlib import Path
from functools import wraps

# 导入 lina_test 中的核心功能
from lina_test import (
    MessageBuffer,
    get_display_data,
    extract_content_string,
)

# 导入 TradingAgents 核心模块
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = Flask(__name__)

# 配置 CORS
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

# 存储正在进行的分析
analyses = {}


class AnalysisRunner:
    """运行 TradingAgents 分析并返回格式化的结果"""

    def __init__(self, analysis_id, config):
        self.analysis_id = analysis_id
        self.config = config
        self.status = 'initializing'
        self.progress = 0
        self.current_stage = 'Initializing analysis...'
        self.result = None
        self.error = None
        self.message_buffer = MessageBuffer()
        self.translate_content = config.get('translateContent', True)
        self.cancelled = False
        self.thread = None

    def update_research_team_status(self, status):
        """Update status for all research team members and trader"""
        research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
        for agent in research_team:
            self.message_buffer.update_agent_status(agent, status)

    def calculate_progress(self):
        """根据智能体状态动态计算进度百分比"""
        if not hasattr(self, 'message_buffer') or not self.message_buffer:
            return self.progress if hasattr(self, 'progress') else 0
        
        # 定义各个团队的权重（总权重为100）
        # 初始化阶段: 10%
        # Analyst Team: 30% (每个分析师平均分配)
        # Research Team: 25% (Bull 5%, Bear 5%, Manager 10%, Trader 5%)
        # Risk Management: 25% (Risky 7%, Safe 7%, Neutral 7%, Portfolio Manager 4%)
        # 最终处理: 10%
        
        progress = 0
        
        # 1. 初始化阶段 (10%)
        if self.status in ['running', 'completed'] and self.current_stage not in ['Initializing analysis...', 'Preparing environment...', 'Initializing TradingAgents graph...']:
            progress += 10
        
        # 2. Analyst Team (30%)
        analyst_value_to_agent = {
            "market": "Market Analyst",
            "social": "Social Analyst",
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst"
        }
        
        selected_agent_names = set()
        if hasattr(self.message_buffer, 'selected_analysts') and self.message_buffer.selected_analysts:
            selected_agent_names = {
                analyst_value_to_agent[analyst]
                for analyst in self.message_buffer.selected_analysts
                if analyst in analyst_value_to_agent
            }
        
        if selected_agent_names:
            analyst_weight_per_agent = 30.0 / len(selected_agent_names)
            for agent_name in selected_agent_names:
                status = self.message_buffer.agent_status.get(agent_name, "pending")
                if status == "completed":
                    progress += analyst_weight_per_agent
                elif status == "in_progress":
                    progress += analyst_weight_per_agent * 0.5  # 进行中算一半
        
        # 3. Research Team (25%)
        research_agents = {
            "Bull Researcher": 5,
            "Bear Researcher": 5,
            "Research Manager": 10,
            "Trader": 5
        }
        
        for agent_name, weight in research_agents.items():
            status = self.message_buffer.agent_status.get(agent_name, "pending")
            if status == "completed":
                progress += weight
            elif status == "in_progress":
                progress += weight * 0.5
        
        # 4. Risk Management Team (25%)
        risk_agents = {
            "Risky Analyst": 7,
            "Safe Analyst": 7,
            "Neutral Analyst": 7,
            "Portfolio Manager": 4
        }
        
        for agent_name, weight in risk_agents.items():
            status = self.message_buffer.agent_status.get(agent_name, "pending")
            if status == "completed":
                progress += weight
            elif status == "in_progress":
                progress += weight * 0.5
        
        # 5. 最终处理阶段 (10%)
        # 如果流式处理完成，添加这部分进度
        if self.status == 'running' and self.current_stage in ['Processing final results...', 'Updating final reports...', 'Translating multi-update sections...', 'Generating display data...']:
            progress += 10
        elif self.status == 'completed':
            progress = 100  # 完成时直接设为100
        
        # 确保进度在 0-100 之间
        return min(100, max(0, int(progress)))

    def cancel(self):
        """取消正在进行的分析"""
        if self.status in ['running', 'initializing']:
            self.cancelled = True
            self.status = 'cancelled'
            self.current_stage = 'Analysis cancelled by user'
            self.progress = 0
            print(f"分析 {self.analysis_id} 已被取消")
            return True
        return False

    def cleanup_chromadb(self):
        """清理 ChromaDB 集合,确保每次分析都使用全新数据,避免数据污染"""
        try:
            import shutil

            print("🧹 清理 ChromaDB 以确保分析独立性...")

            # 方法 1: 删除整个数据目录 (推荐,最彻底)
            chroma_paths = ['./chroma', './chromadb', './chroma_temp']

            cleaned = False
            for path in chroma_paths:
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        print(f"  ✅ 已删除 ChromaDB 目录: {path}")
                        cleaned = True
                    except Exception as e:
                        print(f"  ⚠️  删除失败 {path}: {e}")

            # 方法 2: 如果目录删除失败,尝试删除特定集合
            if not cleaned:
                try:
                    import chromadb

                    # 尝试连接到可能存在的 ChromaDB 实例
                    for path in chroma_paths:
                        if os.path.exists(path):
                            try:
                                client = chromadb.PersistentClient(path=path)
                            except:
                                continue
                        else:
                            try:
                                client = chromadb.Client()
                            except:
                                continue

                        # 删除所有可能的集合 (TradingAgents 使用这五个)
                        collection_names = ['bull_memory', 'bear_memory', 'trader_memory', 'invest_judge_memory', 'risk_manager_memory']
                        for name in collection_names:
                            try:
                                client.delete_collection(name=name)
                                print(f"  ✅ 已删除集合: {name}")
                            except:
                                pass  # 集合可能不存在,忽略错误

                        break  # 成功连接后退出

                except ImportError:
                    print("  ℹ️  ChromaDB 未安装,跳过集合清理")
                except Exception as e:
                    print(f"  ⚠️  集合清理时出现错误: {e}")

            print("✅ ChromaDB 清理完成 - 确保每次分析使用全新数据")

        except Exception as e:
            print(f"⚠️  ChromaDB 清理时出现非关键错误: {e}")
            print("  分析将继续,但可能遇到集合冲突")
            # 不中断流程,让 TradingAgents 自己处理

    def run(self):
        """执行 TradingAgents 分析"""
        try:
            # 检查是否在开始前就被取消
            if self.cancelled:
                self.status = 'cancelled'
                self.current_stage = 'Analysis cancelled by user'
                return
            
            self.status = 'running'
            self.current_stage = 'Preparing environment...'
            self.progress = 5

            # ========== 清理 ChromaDB (避免集合冲突) ==========
            self.cleanup_chromadb()

            # 从配置中获取参数（支持多种格式）
            # 支持 ticker 或 tickers 字段
            ticker = self.config.get('ticker') or (self.config.get('tickers', ['SPY'])[0] if isinstance(self.config.get('tickers'), list) else 'SPY')
            # 支持 analysis_date 或 date 字段
            analysis_date = self.config.get('analysis_date') or self.config.get('date', datetime.now().strftime('%Y-%m-%d'))
            # 支持 analysts 字段（可能是列表或需要转换）
            selected_analysts = self.config.get('analysts', ['market', 'social', 'news', 'fundamentals'])
            # 如果 analysts 是字符串列表，直接使用；如果是对象列表，提取 value
            if selected_analysts and isinstance(selected_analysts[0], dict):
                selected_analysts = [a.get('value', a.get('name', '')) for a in selected_analysts if a.get('value') or a.get('name')]
            # 确保所有 analyst 值都是小写（TradingAgentsGraph 期望小写）
            selected_analysts = [a.lower() if isinstance(a, str) else str(a).lower() for a in selected_analysts]
            # 支持 research_depth 或 researchDepth 字段
            research_depth = self.config.get('research_depth') or self.config.get('researchDepth', 2)
            # 支持 llm_provider 或 llmProvider 字段
            llm_provider = (self.config.get('llm_provider') or self.config.get('llmProvider', 'openai')).lower()
            # 支持 backend_url 或 backendUrl 字段
            backend_url = self.config.get('backend_url') or self.config.get('backendUrl')
            # 支持 shallow_thinker 或 quickThinkLLM 字段
            shallow_thinker = self.config.get('shallow_thinker') or self.config.get('quickThinkLLM', 'gpt-4o-mini')
            # 支持 deep_thinker 或 deepThinkLLM 字段
            deep_thinker = self.config.get('deep_thinker') or self.config.get('deepThinkLLM', 'gpt-4o-mini')

            # 创建配置
            config = DEFAULT_CONFIG.copy()
            config["max_debate_rounds"] = research_depth
            config["max_risk_discuss_rounds"] = research_depth
            config["quick_think_llm"] = shallow_thinker
            config["deep_think_llm"] = deep_thinker
            config["backend_url"] = backend_url
            config["llm_provider"] = llm_provider

            self.current_stage = 'Initializing TradingAgents graph...'
            self.progress = 10
            
            # 存储选择的智能体到 message_buffer（用于进度计算）
            self.message_buffer.selected_analysts = selected_analysts

            # 初始化 graph
            graph = TradingAgentsGraph(
                selected_analysts, config=config, debug=True
            )

            # 创建结果目录
            results_dir = Path(config["results_dir"]) / ticker / analysis_date
            results_dir.mkdir(parents=True, exist_ok=True)
            report_dir = results_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            log_file = results_dir / "message_tool.log"
            log_file.touch(exist_ok=True)

            # 设置消息和工具调用保存装饰器
            def save_message_decorator(obj, func_name):
                func = getattr(obj, func_name)
                @wraps(func)
                def wrapper(*args, **kwargs):
                    func(*args, **kwargs)
                    timestamp, message_type, content = obj.messages[-1]
                    content = content.replace("\n", " ")
                    with open(log_file, "a") as f:
                        f.write(f"{timestamp} [{message_type}] {content}\n")
                return wrapper

            def save_tool_call_decorator(obj, func_name):
                func = getattr(obj, func_name)
                @wraps(func)
                def wrapper(*args, **kwargs):
                    func(*args, **kwargs)
                    timestamp, tool_name, args_data = obj.tool_calls[-1]
                    # 处理 args 可能是字典或其他格式
                    if isinstance(args_data, dict):
                        args_str = ", ".join(f"{k}={v}" for k, v in args_data.items())
                    else:
                        args_str = str(args_data)
                    with open(log_file, "a") as f:
                        f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
                return wrapper

            # 定义会被多次更新的 section（只在最终完成时翻译）
            MULTI_UPDATE_SECTIONS = {"investment_plan", "final_trade_decision"}
            
            # 跟踪正在翻译的section，避免重复翻译
            translating_sections = set()
            section_content_hash = {}  # 跟踪每个section的内容hash，避免相同内容重复翻译
            
            def save_report_section_decorator(obj, func_name):
                func = getattr(obj, func_name)
                @wraps(func)
                def wrapper(section_name, content):
                    func(section_name, content)
                    if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                        content = obj.report_sections[section_name]
                        if content:
                            file_name = f"{section_name}.md"
                            with open(report_dir / file_name, "w") as f:
                                f.write(content)
                            
                            # 如果启用了翻译，且不是会被多次更新的section，立即异步翻译（不阻塞主流程）
                            # 对于会被多次更新的section，在最终完成时统一翻译，避免浪费资源
                            if self.translate_content and content and section_name not in MULTI_UPDATE_SECTIONS:
                                # 计算内容hash，检查内容是否真的改变了
                                import hashlib
                                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                                
                                # 检查是否已经有翻译结果，且内容没有改变
                                if hasattr(obj, 'translated_report_sections') and section_name in obj.translated_report_sections:
                                    existing_translation = obj.translated_report_sections[section_name]
                                    if existing_translation and section_name in section_content_hash and section_content_hash[section_name] == content_hash:
                                        # 内容没有改变，且已有翻译，跳过
                                        return
                                
                                # 检查是否正在翻译中
                                if section_name in translating_sections:
                                    # 正在翻译中，跳过
                                    return
                                
                                # 内容改变了或没有翻译，开始翻译
                                section_content_hash[section_name] = content_hash
                                translating_sections.add(section_name)
                                
                                def translate_section():
                                    try:
                                        from lina_test import translate_to_chinese
                                        # 再次获取最新内容（可能在翻译过程中被更新）
                                        current_content = obj.report_sections.get(section_name)
                                        if current_content:
                                            # 再次检查hash，确保内容没有在翻译过程中改变
                                            current_hash = hashlib.md5(current_content.encode('utf-8')).hexdigest()
                                            if current_hash == content_hash:
                                                translated = translate_to_chinese(current_content, enable_translation=True)
                                                # 存储翻译结果
                                                if hasattr(obj, 'translated_report_sections'):
                                                    obj.translated_report_sections[section_name] = translated
                                                    print(f"✅ 已翻译 {section_name} ({len(current_content)} 字符)")
                                            else:
                                                print(f"⚠️  {section_name} 内容在翻译过程中已更新，跳过此次翻译")
                                    except Exception as e:
                                        print(f"⚠️  翻译 {section_name} 时出错: {e}")
                                        # 翻译失败时，标记为None，后续会实时翻译
                                        if hasattr(obj, 'translated_report_sections'):
                                            obj.translated_report_sections[section_name] = None
                                    finally:
                                        # 翻译完成，从正在翻译的集合中移除
                                        translating_sections.discard(section_name)
                                
                                # 在后台线程中异步翻译
                                translation_thread = threading.Thread(target=translate_section)
                                translation_thread.daemon = True
                                translation_thread.start()
                return wrapper

            self.message_buffer.add_message = save_message_decorator(self.message_buffer, "add_message")
            self.message_buffer.add_tool_call = save_tool_call_decorator(self.message_buffer, "add_tool_call")
            self.message_buffer.update_report_section = save_report_section_decorator(
                self.message_buffer, "update_report_section"
            )

            # 存储选择的代理
            self.message_buffer.selected_analysts = selected_analysts

            # 添加初始消息
            self.message_buffer.add_message("System", f"Selected ticker: {ticker}")
            self.message_buffer.add_message("System", f"Analysis date: {analysis_date}")
            self.message_buffer.add_message(
                "System",
                f"Selected analysts: {', '.join(selected_analysts)}",
            )

            # 重置代理状态
            for agent in self.message_buffer.agent_status:
                self.message_buffer.update_agent_status(agent, "pending")

            # 重置报告部分
            for section in self.message_buffer.report_sections:
                self.message_buffer.report_sections[section] = None
            self.message_buffer.current_report = None
            self.message_buffer.final_report = {}

            # 设置第一个分析师为 in_progress
            first_analyst = f"{selected_analysts[0].capitalize()} Analyst"
            self.message_buffer.update_agent_status(first_analyst, "in_progress")

            self.current_stage = f'Analyzing {ticker}...'
            self.progress = self.calculate_progress()

            # 初始化状态并获取 graph args
            init_agent_state = graph.propagator.create_initial_state(ticker, analysis_date)
            args = graph.propagator.get_graph_args()

            # 流式处理分析
            trace = []
            last_progress_update = 20  # 从20%开始
            for chunk in graph.graph.stream(init_agent_state, **args):
                # 检查是否被取消
                if self.cancelled:
                    self.status = 'cancelled'
                    self.current_stage = 'Analysis cancelled by user'
                    self.message_buffer.add_message("System", "Analysis cancelled by user")
                    print(f"分析 {self.analysis_id} 已被取消，停止处理")
                    return
                
                trace.append(chunk)
                
                if len(chunk.get("messages", [])) > 0:
                    # 获取最后一条消息
                    last_message = chunk["messages"][-1]

                    # 提取消息内容和类型
                    if hasattr(last_message, "content"):
                        content = extract_content_string(last_message.content)
                        msg_type = "Reasoning"
                    else:
                        content = str(last_message)
                        msg_type = "System"

                    # 添加消息到缓冲区
                    self.message_buffer.add_message(msg_type, content)

                    # 如果是工具调用，添加到工具调用列表
                    if hasattr(last_message, "tool_calls"):
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                self.message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                self.message_buffer.add_tool_call(tool_call.name, tool_call.args)

                    # 根据 chunk 内容更新报告和代理状态
                    # Analyst Team Reports
                    if "market_report" in chunk and chunk["market_report"]:
                        self.message_buffer.update_report_section(
                            "market_report", chunk["market_report"]
                        )
                        self.message_buffer.update_agent_status("Market Analyst", "completed")
                        self.progress = self.calculate_progress()
                        if "social" in selected_analysts:
                            self.message_buffer.update_agent_status("Social Analyst", "in_progress")
                            self.progress = self.calculate_progress()

                    if "sentiment_report" in chunk and chunk["sentiment_report"]:
                        self.message_buffer.update_report_section(
                            "sentiment_report", chunk["sentiment_report"]
                        )
                        self.message_buffer.update_agent_status("Social Analyst", "completed")
                        self.progress = self.calculate_progress()
                        if "news" in selected_analysts:
                            self.message_buffer.update_agent_status("News Analyst", "in_progress")
                            self.progress = self.calculate_progress()

                    if "news_report" in chunk and chunk["news_report"]:
                        self.message_buffer.update_report_section(
                            "news_report", chunk["news_report"]
                        )
                        self.message_buffer.update_agent_status("News Analyst", "completed")
                        self.progress = self.calculate_progress()
                        if "fundamentals" in selected_analysts:
                            self.message_buffer.update_agent_status("Fundamentals Analyst", "in_progress")
                            self.progress = self.calculate_progress()

                    if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                        self.message_buffer.update_report_section(
                            "fundamentals_report", chunk["fundamentals_report"]
                        )
                        self.message_buffer.update_agent_status("Fundamentals Analyst", "completed")
                        self.progress = self.calculate_progress()
                        self.update_research_team_status("in_progress")
                        self.progress = self.calculate_progress()

                    # Research Team
                    if "investment_debate_state" in chunk and chunk["investment_debate_state"]:
                        debate_state = chunk["investment_debate_state"]

                        if "bull_history" in debate_state and debate_state["bull_history"]:
                            self.update_research_team_status("in_progress")
                            bull_responses = debate_state["bull_history"].split("\n")
                            latest_bull = bull_responses[-1] if bull_responses else ""
                            if latest_bull:
                                self.message_buffer.add_message("Reasoning", latest_bull)
                                self.message_buffer.update_report_section(
                                    "investment_plan",
                                    f"### Bull Researcher Analysis\n{latest_bull}",
                                )

                        if "bear_history" in debate_state and debate_state["bear_history"]:
                            self.update_research_team_status("in_progress")
                            bear_responses = debate_state["bear_history"].split("\n")
                            latest_bear = bear_responses[-1] if bear_responses else ""
                            if latest_bear:
                                self.message_buffer.add_message("Reasoning", latest_bear)
                                current_plan = self.message_buffer.report_sections.get('investment_plan', '')
                                self.message_buffer.update_report_section(
                                    "investment_plan",
                                    f"{current_plan}\n\n### Bear Researcher Analysis\n{latest_bear}",
                                )

                        if "judge_decision" in debate_state and debate_state["judge_decision"]:
                            self.update_research_team_status("in_progress")
                            self.message_buffer.add_message(
                                "Reasoning",
                                f"Research Manager: {debate_state['judge_decision']}",
                            )
                            current_plan = self.message_buffer.report_sections.get('investment_plan', '')
                            self.message_buffer.update_report_section(
                                "investment_plan",
                                f"{current_plan}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                            )
                            # 更新 Research Team 状态
                            self.message_buffer.update_agent_status("Bull Researcher", "completed")
                            self.message_buffer.update_agent_status("Bear Researcher", "completed")
                            self.message_buffer.update_agent_status("Research Manager", "completed")
                            self.progress = self.calculate_progress()
                            self.message_buffer.update_agent_status("Risky Analyst", "in_progress")
                            self.progress = self.calculate_progress()

                    # Trading Team
                    if "trader_investment_plan" in chunk and chunk["trader_investment_plan"]:
                        self.message_buffer.update_report_section(
                            "trader_investment_plan", chunk["trader_investment_plan"]
                        )
                        self.message_buffer.update_agent_status("Trader", "completed")
                        self.progress = self.calculate_progress()
                        self.message_buffer.update_agent_status("Risky Analyst", "in_progress")
                        self.progress = self.calculate_progress()

                    # Risk Management Team
                    if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                        risk_state = chunk["risk_debate_state"]

                        if "current_risky_response" in risk_state and risk_state["current_risky_response"]:
                            self.message_buffer.update_agent_status("Risky Analyst", "in_progress")
                            self.message_buffer.add_message(
                                "Reasoning",
                                f"Risky Analyst: {risk_state['current_risky_response']}",
                            )
                            self.message_buffer.update_report_section(
                                "final_trade_decision",
                                f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                            )

                        if "current_safe_response" in risk_state and risk_state["current_safe_response"]:
                            self.message_buffer.update_agent_status("Safe Analyst", "in_progress")
                            self.message_buffer.add_message(
                                "Reasoning",
                                f"Safe Analyst: {risk_state['current_safe_response']}",
                            )
                            self.message_buffer.update_report_section(
                                "final_trade_decision",
                                f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                            )

                        if "current_neutral_response" in risk_state and risk_state["current_neutral_response"]:
                            self.message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                            self.message_buffer.add_message(
                                "Reasoning",
                                f"Neutral Analyst: {risk_state['current_neutral_response']}",
                            )
                            self.message_buffer.update_report_section(
                                "final_trade_decision",
                                f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                            )

                        if "judge_decision" in risk_state and risk_state["judge_decision"]:
                            self.message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                            self.message_buffer.add_message(
                                "Reasoning",
                                f"Portfolio Manager: {risk_state['judge_decision']}",
                            )
                            self.message_buffer.update_report_section(
                                "final_trade_decision",
                                f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                            )
                            self.message_buffer.update_agent_status("Risky Analyst", "completed")
                            self.message_buffer.update_agent_status("Safe Analyst", "completed")
                            self.message_buffer.update_agent_status("Neutral Analyst", "completed")
                            self.message_buffer.update_agent_status("Portfolio Manager", "completed")
                            self.progress = self.calculate_progress()  # 根据智能体状态动态计算进度

            # 流式处理完成，更新进度和状态
            self.current_stage = 'Processing final results...'
            self.progress = self.calculate_progress()

            # 获取最终状态和决策
            final_state = trace[-1]
            decision = graph.process_signal(final_state["final_trade_decision"])

            # 更新代理状态
            analyst_value_to_agent = {
                "market": "Market Analyst",
                "social": "Social Analyst",
                "news": "News Analyst",
                "fundamentals": "Fundamentals Analyst"
            }
            selected_agent_names = set()
            if self.message_buffer.selected_analysts:
                selected_agent_names = {
                    analyst_value_to_agent[analyst]
                    for analyst in self.message_buffer.selected_analysts
                    if analyst in analyst_value_to_agent
                }

            for agent in self.message_buffer.agent_status:
                if agent in ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"]:
                    if agent in selected_agent_names:
                        self.message_buffer.update_agent_status(agent, "completed")
                else:
                    if agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                        if self.message_buffer.report_sections.get("investment_plan"):
                            self.message_buffer.update_agent_status(agent, "completed")
                    elif agent == "Trader":
                        if self.message_buffer.report_sections.get("trader_investment_plan"):
                            self.message_buffer.update_agent_status(agent, "completed")
                    elif agent in ["Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]:
                        if self.message_buffer.report_sections.get("final_trade_decision"):
                            self.message_buffer.update_agent_status(agent, "completed")

            self.progress = self.calculate_progress()
            self.message_buffer.add_message(
                "Analysis", f"Completed analysis for {analysis_date}"
            )

            # 更新最终报告部分
            self.current_stage = 'Updating final reports...'
            self.progress = self.calculate_progress()
            for section in self.message_buffer.report_sections.keys():
                if section in final_state:
                    self.message_buffer.update_report_section(section, final_state[section])

            # 确保 final_report 已更新
            self.message_buffer._update_final_report()

            # 对会被多次更新的 section 进行最终翻译（避免浪费资源）
            if self.translate_content:
                self.current_stage = 'Translating multi-update sections...'
                self.progress = self.calculate_progress()
                print(f"开始翻译会被多次更新的 section（investment_plan, final_trade_decision）...")
                
                def translate_multi_update_sections():
                    """翻译会被多次更新的 section"""
                    from lina_test import translate_to_chinese
                    for section_name in MULTI_UPDATE_SECTIONS:
                        content = self.message_buffer.report_sections.get(section_name)
                        if content:
                            try:
                                translated = translate_to_chinese(content, enable_translation=True)
                                if hasattr(self.message_buffer, 'translated_report_sections'):
                                    self.message_buffer.translated_report_sections[section_name] = translated
                                    print(f"✅ 已翻译 {section_name} ({len(content)} 字符)")
                            except Exception as e:
                                print(f"⚠️  翻译 {section_name} 时出错: {e}")
                                if hasattr(self.message_buffer, 'translated_report_sections'):
                                    self.message_buffer.translated_report_sections[section_name] = None
                
                # 在后台线程中异步翻译（不阻塞主流程）
                translation_thread = threading.Thread(target=translate_multi_update_sections)
                translation_thread.daemon = True
                translation_thread.start()
                
                # 等待翻译完成（最多等待30秒）
                import time
                max_wait_time = 30
                wait_interval = 0.5
                waited_time = 0
                
                # 检查哪些 section 需要翻译（有内容的）
                sections_to_translate = [
                    section_name for section_name in MULTI_UPDATE_SECTIONS
                    if self.message_buffer.report_sections.get(section_name)
                ]
                
                if sections_to_translate:
                    while waited_time < max_wait_time:
                        all_translated = True
                        for section_name in sections_to_translate:
                            translated = self.message_buffer.translated_report_sections.get(section_name)
                            if translated is None:
                                all_translated = False
                                break
                        
                        if all_translated:
                            break
                        
                        time.sleep(wait_interval)
                        waited_time += wait_interval
                    
                    if waited_time >= max_wait_time:
                        print(f"⚠️  翻译等待超时，使用已翻译的内容和原文")
                else:
                    print(f"ℹ️  没有需要翻译的多更新 section")

            # 生成显示数据（优先使用已翻译的内容）
            self.current_stage = 'Generating display data...'
            self.progress = self.calculate_progress()
            print(f"开始生成显示数据（优先使用已翻译的内容）...")

            # 获取最终显示数据（优先使用已翻译的内容，如果没有则实时翻译）
            # 注意：翻译是异步进行的，如果某些section的翻译还在进行中，
            # 前端可以通过轮询status接口来获取最新的翻译结果
            self.result = get_display_data(
                translate_content=self.translate_content,
                message_buffer_instance=self.message_buffer
            )

            # 只有在所有处理完成后才设置为completed
            self.status = 'completed'
            self.current_stage = 'Analysis completed'
            self.progress = 100
            print(f"分析 {self.analysis_id} 完全完成")

        except Exception as e:
            # 如果被取消，不标记为失败
            if self.cancelled:
                self.status = 'cancelled'
                self.current_stage = 'Analysis cancelled by user'
                return
            self.status = 'failed'
            self.error = f'Analysis failed: {str(e)}'
            print(f"分析失败: {e}")
            traceback.print_exc()


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        tradingagents_available = True
    except:
        tradingagents_available = False

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tradingagents_available': tradingagents_available,
        'version': 'lina-server-v1.0'
    })


@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """启动新的分析"""
    try:
        config = request.json

        # 验证必需字段
        if not config.get('ticker'):
            return jsonify({'error': 'ticker is required'}), 400

        # 生成唯一分析 ID
        analysis_id = str(uuid.uuid4())

        # 创建并启动分析运行器
        runner = AnalysisRunner(analysis_id, config)
        analyses[analysis_id] = runner

        # 在后台线程中运行分析
        thread = threading.Thread(target=runner.run)
        thread.daemon = True
        runner.thread = thread  # 保存线程引用以便取消
        thread.start()

        return jsonify({
            'success': True,
            'analysisId': analysis_id,
            'message': 'Analysis started'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/<analysis_id>/status', methods=['GET'])
def get_analysis_status(analysis_id):
    """获取正在进行的分析状态"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    runner = analyses[analysis_id]

    # 获取当前的显示数据（用于实时更新）
    display_data = None
    if runner.message_buffer:
        try:
            display_data = get_display_data(
                translate_content=runner.translate_content,
                message_buffer_instance=runner.message_buffer
            )
        except:
            pass

    return jsonify({
        'analysisId': analysis_id,
        'status': runner.status,
        'progress': runner.progress,
        'currentStage': runner.current_stage,
        'error': runner.error,
        'displayData': display_data  # 实时返回显示数据
    })


@app.route('/api/analysis/<analysis_id>/results', methods=['GET'])
def get_analysis_results(analysis_id):
    """获取已完成分析的结果"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    runner = analyses[analysis_id]

    if runner.status != 'completed':
        return jsonify({
            'error': 'Analysis not completed',
            'status': runner.status
        }), 400

    return jsonify({
        'analysisId': analysis_id,
        'result': runner.result
    })


@app.route('/api/analysis/<analysis_id>/cancel', methods=['POST'])
def cancel_analysis(analysis_id):
    """取消正在进行的分析"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    runner = analyses[analysis_id]

    # 尝试取消分析
    if runner.cancel():
        # 从 analyses 字典中移除（可选，取决于是否要保留历史记录）
        # del analyses[analysis_id]
        return jsonify({
            'success': True,
            'analysisId': analysis_id,
            'message': 'Analysis cancelled successfully',
            'status': runner.status
        })
    else:
        return jsonify({
            'success': False,
            'analysisId': analysis_id,
            'message': f'Cannot cancel analysis. Current status: {runner.status}',
            'status': runner.status
        }), 400


@app.route('/api/analysis', methods=['GET'])
def list_analyses():
    """列出所有分析"""
    return jsonify({
        'analyses': [
            {
                'analysisId': aid,
                'status': runner.status,
                'config': runner.config,
                'timestamp': datetime.now().isoformat()
            }
            for aid, runner in analyses.items()
        ]
    })


if __name__ == '__main__':
    import os

    # 从环境变量读取配置
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    print("=" * 60)
    print("TradingAgents API Server (Lina)")
    print("=" * 60)
    print(f"🚀 Starting server on {host}:{port}")
    print(f"   Debug mode: {debug}")
    print(f"   Environment: {'Development' if debug else 'Production'}")
    print("\nMake sure environment variables are set:")
    print("  ✅ OPENAI_API_KEY")
    print("  ✅ ALPHA_VANTAGE_API_KEY (optional)")
    print("=" * 60)

    # 运行 Flask 应用
    app.run(host=host, port=port, debug=debug, threaded=True)

