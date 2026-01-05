from typing import Optional
import datetime
import json
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
from rich import box
from rich.align import Align

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.utils import *

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = {}  # Store the complete final report as a dictionary
        self.selected_analysts = []  # Store user-selected analysts (e.g., ["market", "social"])
        self.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }
        # 存储已翻译的报告内容（用于流式翻译）
        self.translated_report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        # 重新构建完整的报告字典（包含所有有内容的 sections）
        self.final_report = {}

        # Analyst Team Reports
        analyst_reports = {}
        if self.report_sections["market_report"]:
            analyst_reports["Market Analyst"] = self.report_sections["market_report"]
        if self.report_sections["sentiment_report"]:
            analyst_reports["Social Analyst"] = self.report_sections["sentiment_report"]
        if self.report_sections["news_report"]:
            analyst_reports["News Analyst"] = self.report_sections["news_report"]
        if self.report_sections["fundamentals_report"]:
            analyst_reports["Fundamentals Analyst"] = self.report_sections["fundamentals_report"]
        
        if analyst_reports:
            self.final_report["Analyst Team"] = analyst_reports

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            self.final_report["Research Team"] = {
                "investment_plan": self.report_sections["investment_plan"]
            }

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            self.final_report["Trading Team"] = {
                "trader_investment_plan": self.report_sections["trader_investment_plan"]
            }

        # Portfolio Management Decision
        if self.report_sections["final_trade_decision"]:
            self.final_report["Portfolio Management"] = {
                "final_trade_decision": self.report_sections["final_trade_decision"]
            }


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def translate_to_chinese(text, enable_translation=True):
    """
    将英文文本翻译为中文
    
    Args:
        text: 要翻译的英文文本
        enable_translation: 是否启用翻译（默认 True）
    
    Returns:
        str: 翻译后的中文文本，如果翻译失败或未启用则返回原文
    """
    if not enable_translation or not text:
        return text
    
    try:
        from openai import OpenAI
        import os
        import re
        
        # 检查是否有 OpenAI API Key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return text  # 如果没有 API Key，返回原文
        
        client = OpenAI(
            api_key=api_key,
            timeout=300.0,  # 增加超时时间以处理长文本
            max_retries=2
        )
        
        # 移除字数限制，翻译完整文本
        # 如果文本非常长（超过模型上下文限制），分批处理
        # GPT-4o-mini 的上下文窗口约为 128K tokens，但为了安全，我们分批处理超过 100K 字符的文本
        max_chunk_size = 100000  # 每批最多 100K 字符
        
        if len(text) <= max_chunk_size:
            # 文本不长，直接翻译
            text_to_translate = text
            # 根据文本长度动态设置 max_tokens（约为文本长度的 1.5 倍，但不超过 16384）
            estimated_tokens = len(text) // 3  # 粗略估计：1 token ≈ 3-4 字符
            max_tokens = min(int(estimated_tokens * 1.5), 16384)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的金融翻译专家。

你的任务是将基于英文信息源的股票分析报告翻译成中文。

注意:
- 保持所有数据、比例、百分比不变
- 保留专业术语的准确性
- 公司名称可以用中文+英文格式,如: 高盛集团(Goldman Sachs)
- 保留 BUY/SELL/HOLD 等关键词,可在后面加中文注释
- 输出纯文本,不要使用 Markdown 格式符号(如 **, ##, ###)

翻译要求:
- 自然流畅的中文表达
- 保持原文的逻辑结构
- 专业、准确、易读
- 纯文本输出,不带格式符号
- 完整翻译所有内容，不要截断"""
                    },
                    {
                        "role": "user",
                        "content": f"请将以下基于英文信息源的股票分析完整翻译成中文(输出纯文本,不要 Markdown 格式,不要截断任何内容):\n\n{text_to_translate}"
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            translated = response.choices[0].message.content
        else:
            # 文本很长，分批翻译
            translated_parts = []
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            for i, chunk in enumerate(chunks):
                estimated_tokens = len(chunk) // 3
                max_tokens = min(int(estimated_tokens * 1.5), 16384)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """你是一个专业的金融翻译专家。

你的任务是将基于英文信息源的股票分析报告翻译成中文。

注意:
- 保持所有数据、比例、百分比不变
- 保留专业术语的准确性
- 公司名称可以用中文+英文格式,如: 高盛集团(Goldman Sachs)
- 保留 BUY/SELL/HOLD 等关键词,可在后面加中文注释
- 输出纯文本,不要使用 Markdown 格式符号(如 **, ##, ###)

翻译要求:
- 自然流畅的中文表达
- 保持原文的逻辑结构
- 专业、准确、易读
- 纯文本输出,不带格式符号
- 完整翻译所有内容，不要截断"""
                        },
                        {
                            "role": "user",
                            "content": f"请将以下基于英文信息源的股票分析完整翻译成中文(输出纯文本,不要 Markdown 格式,不要截断任何内容):\n\n{chunk}"
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                
                translated_parts.append(response.choices[0].message.content)
            
            translated = "\n\n".join(translated_parts)
        
        # 清理可能残留的 Markdown 格式
        translated = re.sub(r'^#{1,6}\s+', '', translated, flags=re.MULTILINE)
        translated = re.sub(r'\*\*(.+?)\*\*', r'\1', translated)
        translated = re.sub(r'`(.+?)`', r'\1', translated)
        translated = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', translated)
        translated = re.sub(r'\n{3,}', '\n\n', translated)
        
        return translated.strip()
        
    except Exception as e:
        # 翻译失败时返回原文
        return text


def get_display_data(spinner_text=None, translate_content=True, message_buffer_instance=None):
    """
    获取前端可用的显示数据（层级嵌套的 JSON 格式）
    
    返回一个字典，包含：
    - 头部信息
    - 团队和代理的状态及报告（层级嵌套）
    - 分析报告数据（按 section 组织）
    - 统计信息
    
    注意：此函数只返回结果数据，不包含过程数据（messages）。
    
    Args:
        spinner_text: 可选的 spinner 文本（当前未使用）
        translate_content: 是否翻译报告内容为中文（默认 True）
                          - True: 翻译 LLM 生成的报告内容为中文，但 API key 保持英文
                          - False: 报告内容保持英文，API key 也保持英文
                          需要设置 OPENAI_API_KEY 环境变量才能使用内容翻译功能
        message_buffer_instance: 可选的 MessageBuffer 实例（用于服务器模式）
                                 如果为 None，使用全局的 message_buffer（CLI 模式）
    
    Returns:
        dict: 包含所有显示数据的字典，API key 始终为英文，内容根据 translate_content 参数决定是否翻译
        
    Example:
        # 获取字典格式的数据（默认翻译内容）
        data = get_display_data()
        
        # 获取数据但不翻译内容
        data = get_display_data(translate_content=False)
        
        # 在 Flask/FastAPI 中使用（服务器模式）
        data = get_display_data(translate_content=True, message_buffer_instance=runner.message_buffer)
        
        # 在 Flask/FastAPI 中使用
        from flask import jsonify
        return jsonify(get_display_data())
    """
    # 使用提供的 message_buffer 或全局的 message_buffer
    mb = message_buffer_instance if message_buffer_instance is not None else message_buffer
    # API key 始终使用英文，不翻译
    # 字段名映射（始终使用英文 key）
    field_names = {
        "header_title": "title",
        "header_content": "content",
        "header_border_style": "border_style",
        "status": "status",
        "status_value": "status_value",
        "in_progress": "is_in_progress",
        "completed": "is_completed",
        "pending": "is_pending",
        "error": "is_error",
        "not_selected": "is_not_selected",
        "report": "report",
        "has_report": "has_report",
        "agents": "agents",
        "total_agents": "total_agents",
        "completed_count": "completed_count",
        "in_progress_count": "in_progress_count",
        "pending_count": "pending_count",
        "content": "content",
        "has_content": "has_content",
        "content_length": "content_length",
        "current_report": "current_report",
        "final_report": "final_report",
        "report_sections": "sections",
        "waiting": "is_waiting",
        "waiting_message": "waiting_message",
        "sections_count": "sections_count",
        "total_sections": "total_sections",
        "tool_calls": "tool_calls",
        "llm_calls": "llm_calls",
        "generated_reports": "generated_reports",
        "stats_text": "stats_text",
        "header": "header",
        "teams": "teams",
        "analysis": "analysis",
        "footer": "footer",
        "timestamp": "timestamp"
    }
    
    # 状态值映射（用于显示状态文本，但 key 保持英文）
    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
        "error": "error",
        "not_selected": "not_selected"
    }
    
    # 辅助函数：获取状态显示文本（如果需要翻译状态值，可以在这里处理）
    def get_status_display(status):
        return status_map.get(status, status)
    
    # Header 信息（key 使用英文，内容根据 translate_content 决定）
    header_data = {
        field_names["header_title"]: "Welcome to TradingAgents",  # Header 文本保持英文，不翻译
        field_names["header_content"]: "Welcome to TradingAgents CLI\n© Tauric Research (https://github.com/TauricResearch)",
        field_names["header_border_style"]: "green"
    }
    
    # 团队和代理的映射关系
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }
    
    # 代理名称到报告 section 的映射
    agent_to_section = {
        "Market Analyst": "market_report",
        "Social Analyst": "sentiment_report",
        "News Analyst": "news_report",
        "Fundamentals Analyst": "fundamentals_report",
        "Bull Researcher": "investment_plan",  # 部分内容
        "Bear Researcher": "investment_plan",  # 部分内容
        "Research Manager": "investment_plan",  # 部分内容
        "Trader": "trader_investment_plan",
        "Risky Analyst": "final_trade_decision",  # 部分内容
        "Neutral Analyst": "final_trade_decision",  # 部分内容
        "Safe Analyst": "final_trade_decision",  # 部分内容
        "Portfolio Manager": "final_trade_decision",  # 部分内容
    }
    
    # Analyst 值到代理名称的映射（用于过滤）
    analyst_value_to_agent = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst"
    }
    
    # 获取用户选择的代理名称列表（仅针对 Analyst Team）
    selected_agent_names = set()
    if hasattr(mb, 'selected_analysts') and mb.selected_analysts:
        selected_agent_names = {
            analyst_value_to_agent[analyst] 
            for analyst in mb.selected_analysts 
            if analyst in analyst_value_to_agent
        }
    
    # 构建层级嵌套的团队和代理数据
    teams_data = {}
    for team_name, agents in teams.items():
        team_agents = {}
        for agent_name in agents:
            # 检查代理是否被选择（仅针对 Analyst Team）
            is_selected = True
            if team_name == "Analyst Team":
                if selected_agent_names:
                    is_selected = agent_name in selected_agent_names
            
            # 如果未选择，设置状态为 "not_selected"
            if not is_selected:
                status = "not_selected"
                report_content = None
            else:
                # 对于其他团队或已选择的代理，使用实际状态
                status = mb.agent_status.get(agent_name, "pending")
                
                # 获取代理的报告内容
                report_content = None
                section_name = agent_to_section.get(agent_name)
                if section_name and section_name in mb.report_sections:
                    report_content = mb.report_sections[section_name]
            
            # 代理名称作为 key 保持英文，不翻译
            # 状态值保持英文
            status_display = get_status_display(status)
            
            # 翻译报告内容（LLM 生成的文本）
            # 优先使用已翻译的内容（如果存在），否则实时翻译
            translated_report = None
            if report_content:
                if translate_content:
                    # 检查是否有已翻译的内容
                    if hasattr(mb, 'translated_report_sections') and section_name and section_name in mb.translated_report_sections:
                        translated_report = mb.translated_report_sections[section_name]
                        # 如果已翻译内容为空或None，则实时翻译
                        if not translated_report:
                            translated_report = translate_to_chinese(report_content, enable_translation=translate_content)
                    else:
                        # 没有已翻译内容，实时翻译
                        translated_report = translate_to_chinese(report_content, enable_translation=translate_content)
                else:
                    translated_report = report_content
            else:
                translated_report = None
            
            # 使用英文代理名称作为 key
            team_agents[agent_name] = {
                field_names["status"]: status_display,
                field_names["status_value"]: status,
                field_names["in_progress"]: status == "in_progress",
                field_names["completed"]: status == "completed",
                field_names["pending"]: status == "pending",
                field_names["error"]: status == "error",
                field_names["not_selected"]: status == "not_selected",
                field_names["report"]: translated_report,
                field_names["has_report"]: report_content is not None
            }
        
        # 团队名称作为 key 保持英文，不翻译
        team_name_display = team_name
        
        # 对于 Analyst Team，只统计用户选择的代理（排除 not_selected 的）
        if team_name == "Analyst Team" and selected_agent_names:
            agents_for_stats = [agent for agent in agents if agent in selected_agent_names]
        else:
            agents_for_stats = agents
        
        # 计算统计信息（排除 not_selected 的代理）
        completed_count = sum(1 for agent in agents_for_stats if mb.agent_status.get(agent) == "completed")
        in_progress_count = sum(1 for agent in agents_for_stats if mb.agent_status.get(agent) == "in_progress")
        pending_count = sum(1 for agent in agents_for_stats if mb.agent_status.get(agent) == "pending")
        not_selected_count = sum(1 for agent in agents if (team_name == "Analyst Team" and selected_agent_names and agent not in selected_agent_names))
        
        teams_data[team_name_display] = {
            field_names["agents"]: team_agents,
            field_names["total_agents"]: len(agents),  # 显示所有代理的总数
            field_names["completed_count"]: completed_count,
            field_names["in_progress_count"]: in_progress_count,
            field_names["pending_count"]: pending_count,
            field_names["not_selected"]: not_selected_count,  # 未选择的代理数量
        }
    
    # Analysis 报告数据 - 按 section 组织（section 名称作为 key 保持英文）
    report_sections_data = {}
    for section_name, content in mb.report_sections.items():
        # section 名称作为 key 保持英文，不翻译
        if content is not None:
            # 翻译报告内容（LLM 生成的文本）
            # 优先使用已翻译的内容（如果存在），否则实时翻译
            if translate_content:
                # 检查是否有已翻译的内容
                if hasattr(mb, 'translated_report_sections') and section_name in mb.translated_report_sections:
                    translated_content = mb.translated_report_sections[section_name]
                    # 如果已翻译内容为空或None，则实时翻译
                    if not translated_content:
                        translated_content = translate_to_chinese(content, enable_translation=translate_content)
                else:
                    # 没有已翻译内容，实时翻译
                    translated_content = translate_to_chinese(content, enable_translation=translate_content)
            else:
                translated_content = content
            report_sections_data[section_name] = {
                field_names["content"]: translated_content,
                field_names["has_content"]: True,
                field_names["content_length"]: len(content) if content else 0
            }
        else:
            report_sections_data[section_name] = {
                field_names["content"]: None,
                field_names["has_content"]: False,
                field_names["content_length"]: 0
            }
    
    # 处理 final_report：团队名称、代理名称、section 名称作为 key 保持英文，只翻译报告内容
    final_report_display = {}
    if mb.final_report:
        for team_name, team_data in mb.final_report.items():
            # 团队名称作为 key 保持英文，不翻译
            if isinstance(team_data, dict):
                # 如果是字典，key 保持英文，只翻译值（报告内容）
                team_data_display = {}
                for key, value in team_data.items():
                    # key 保持英文，不翻译
                    # 如果值是字符串（报告内容），根据 translate_content 决定是否翻译
                    if isinstance(value, str) and translate_content:
                        # 尝试找到对应的section名称来获取已翻译内容
                        # Analyst Team: key 是 agent name，需要找到对应的 section
                        section_name = None
                        if team_name == "Analyst Team":
                            agent_to_section_map = {
                                "Market Analyst": "market_report",
                                "Social Analyst": "sentiment_report",
                                "News Analyst": "news_report",
                                "Fundamentals Analyst": "fundamentals_report"
                            }
                            section_name = agent_to_section_map.get(key)
                        elif team_name == "Research Team" and key == "investment_plan":
                            section_name = "investment_plan"
                        elif team_name == "Trading Team" and key == "trader_investment_plan":
                            section_name = "trader_investment_plan"
                        elif team_name == "Portfolio Management" and key == "final_trade_decision":
                            section_name = "final_trade_decision"
                        
                        # 优先使用已翻译的内容
                        if section_name and hasattr(mb, 'translated_report_sections') and section_name in mb.translated_report_sections:
                            translated_value = mb.translated_report_sections[section_name]
                            if translated_value:
                                team_data_display[key] = translated_value
                            else:
                                team_data_display[key] = translate_to_chinese(value, enable_translation=translate_content)
                        else:
                            team_data_display[key] = translate_to_chinese(value, enable_translation=translate_content)
                    else:
                        team_data_display[key] = value
                final_report_display[team_name] = team_data_display
            else:
                # 如果不是字典，直接翻译值（如果是字符串）
                if isinstance(team_data, str) and translate_content:
                    final_report_display[team_name] = translate_to_chinese(team_data, enable_translation=translate_content)
                else:
                    final_report_display[team_name] = team_data
    
    # 优先使用 final_report（完整报告，字典格式），如果没有则使用 current_report（当前部分，字符串格式）
    report_content = None
    if mb.final_report and len(mb.final_report) > 0:
        report_content = final_report_display  # 使用处理后的字典格式
    elif mb.current_report:
        # 翻译 current_report（如果是字符串）
        if isinstance(mb.current_report, str) and translate_content:
            report_content = translate_to_chinese(mb.current_report, enable_translation=translate_content)
        else:
            report_content = mb.current_report  # 字符串格式
    
    # 判断是否在等待：检查 final_report_display 是否有实际内容，或 report_sections 是否有内容
    has_final_report_content = False
    if final_report_display:
        # 检查 final_report_display 中是否有任何非空内容
        for team_data in final_report_display.values():
            if isinstance(team_data, dict):
                for value in team_data.values():
                    if value and (isinstance(value, str) and len(value.strip()) > 0):
                        has_final_report_content = True
                        break
                if has_final_report_content:
                    break
            elif isinstance(team_data, str) and len(team_data.strip()) > 0:
                has_final_report_content = True
                break
    
    # 检查 report_sections 是否有内容
    has_sections_content = any(
        s.get(field_names["has_content"], False) 
        for s in report_sections_data.values()
    )
    
    # 如果 final_report 或 report_sections 有内容，则不在等待
    is_waiting = not (has_final_report_content or has_sections_content)
    
    # 系统消息保持英文，不翻译
    waiting_message = "Waiting for analysis report..."
    
    analysis_data = {
        field_names["current_report"]: report_content,  # 可能是字典或字符串
        field_names["final_report"]: final_report_display,  # 字典格式，按团队组织
        field_names["report_sections"]: report_sections_data,
        field_names["waiting"]: is_waiting,
        field_names["waiting_message"]: waiting_message,
        field_names["sections_count"]: sum(1 for s in report_sections_data.values() if s.get(field_names["has_content"], False)),
        field_names["total_sections"]: len(report_sections_data)
    }
    
    # Footer 统计信息（系统消息保持英文，不翻译）
    tool_calls_count = len(mb.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in mb.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in mb.report_sections.values() if content is not None
    )
    
    stats_text = f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    
    footer_data = {
        field_names["tool_calls"]: tool_calls_count,
        field_names["llm_calls"]: llm_calls_count,
        field_names["generated_reports"]: reports_count,
        field_names["stats_text"]: stats_text
    }
    
    # 返回层级嵌套的 JSON 数据结构（不包含过程数据 messages）
    return {
        field_names["header"]: header_data,
        field_names["teams"]: teams_data,
        field_names["analysis"]: analysis_data,
        field_names["footer"]: footer_data,
        field_names["timestamp"]: datetime.datetime.now().isoformat()
    }


def get_display_json(spinner_text=None, indent=None, translate_content=True):
    """
    获取前端可用的 JSON 格式数据
    返回 JSON 字符串，可以直接用于 API 响应
    
    Args:
        spinner_text: 可选的 spinner 文本
        indent: JSON 缩进级别（None 表示紧凑格式，2 表示格式化输出）
        translate_content: 是否翻译报告内容为中文（默认 True）
                          - True: 翻译 LLM 生成的报告内容为中文，但 API key 保持英文
                          - False: 报告内容保持英文，API key 也保持英文
                          需要设置 OPENAI_API_KEY 环境变量才能使用内容翻译功能
    
    Returns:
        str: JSON 格式的字符串
    """
    data = get_display_data(spinner_text, translate_content=translate_content)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: OpenAI backend
    console.print(
        create_question_box(
            "Step 5: OpenAI backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Portfolio Manager Decision
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()

    # Initialize the graph
    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

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
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        # Store selected analysts for filtering display
        message_buffer.selected_analysts = [analyst.value for analyst in selections['analysts']]
        
        # Reset agent statuses
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        # Reset report sections
        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = {}  # Reset to empty dictionary

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        # Initialize state and get graph args
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # Add message to buffer
                message_buffer.add_message(msg_type, content)                

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    # Set next analyst to in_progress
                    if "fundamentals" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Fundamentals Analyst", "in_progress"
                        )

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    # Set all research team members to in_progress
                    update_research_team_status("in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            # Update research report with bull's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            # Update research report with bear's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        # Keep all research team members in progress until final decision
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        # Update research report with final decision
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        # Mark all research team members as completed
                        update_research_team_status("completed")
                        # Set first risk analyst to in_progress
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final decision
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final decision only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                # Update the display
                update_display(layout)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update agent statuses to completed - only for agents that actually ran
        # For Analyst Team, only update selected analysts
        analyst_value_to_agent = {
            "market": "Market Analyst",
            "social": "Social Analyst",
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst"
        }
        selected_agent_names = set()
        if message_buffer.selected_analysts:
            selected_agent_names = {
                analyst_value_to_agent[analyst] 
                for analyst in message_buffer.selected_analysts 
                if analyst in analyst_value_to_agent
            }
        
        # Only update status for agents that actually ran
        # Analyst Team: only selected ones
        # Other teams: all of them (they always run)
        for agent in message_buffer.agent_status:
            if agent in ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"]:
                # Only update if this analyst was selected
                if agent in selected_agent_names:
                    message_buffer.update_agent_status(agent, "completed")
                # Otherwise, keep it as "pending" (don't change)
            else:
                # For other teams, always mark as completed if they have content
                # Check if they have a report section or were actually used
                if agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                    if message_buffer.report_sections.get("investment_plan"):
                        message_buffer.update_agent_status(agent, "completed")
                elif agent == "Trader":
                    if message_buffer.report_sections.get("trader_investment_plan"):
                        message_buffer.update_agent_status(agent, "completed")
                elif agent in ["Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]:
                    if message_buffer.report_sections.get("final_trade_decision"):
                        message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])
        
        # Ensure final_report is updated with all sections
        message_buffer._update_final_report()

        # Display the complete final report
        display_complete_report(final_state)

        update_display(layout)
        
        # Get final display data (will use final_report if available)
        display_data = get_display_data()
        
        # Print display_data as formatted JSON
        console.print("\n[bold green]Display Data (JSON):[/bold green]")
        console.print(json.dumps(display_data, indent=2, ensure_ascii=False))
        
        # Return display_data
        return display_data


@app.command()
def analyze():
    run_analysis()


if __name__ == "__main__":
    app()
