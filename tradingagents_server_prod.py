"""
TradingAgents Frontend API Server - 真实版本
This Flask server connects the web frontend to the actual TradingAgents framework
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
from datetime import datetime
import uuid
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Store for ongoing analyses
analyses = {}

class AnalysisRunner:
    """运行真实的 TradingAgents 分析"""

    def __init__(self, analysis_id, config):
        self.analysis_id = analysis_id
        self.config = config
        self.status = 'initializing'
        self.progress = 0
        self.current_stage = '准备分析...'
        self.result = None
        self.error = None
        # 始终启用中文翻译 (可以通过 config['enableChinese'] = False 禁用)
        self.enable_translation = True if config.get('enableChinese') != False else False
        print(f"🌐 中文翻译: {'启用' if self.enable_translation else '禁用'}")

    def translate_to_chinese(self, text):
        """将英文分析结果翻译为中文 - 仅翻译输出,保留分析的准确性"""
        if not self.enable_translation:
            return text

        try:
            from openai import OpenAI
            import os

            # 创建 OpenAI 客户端
            # 翻译是相对简单的任务,5 分钟足够
            # (默认是 10 分钟,这里适度降低以避免无限等待)
            client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                timeout=300.0,  # 5 分钟 (翻译任务通常 10-30 秒)
                max_retries=2
            )

            print("🌐 正在将英文分析结果翻译为中文...")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的金融翻译专家。

你的任务是将基于英文信息源的股票分析报告翻译成中文。

注意:
- 这些分析是基于英文搜索和数据源的,信息更全面准确
- 保持所有数据、比例、百分比不变
- 保留专业术语的准确性
- 公司名称可以用中文+英文格式,如: 高盛集团(Goldman Sachs)
- 保留 BUY/SELL/HOLD 等关键词,可在后面加中文注释

翻译要求:
- 自然流畅的中文表达
- 保持原文的逻辑结构
- 专业、准确、易读"""
                    },
                    {
                        "role": "user",
                        "content": f"请将以下基于英文信息源的股票分析翻译成中文:\n\n{text[:3000]}"
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )

            translated = response.choices[0].message.content
            print("✅ 翻译完成")
            return translated

        except Exception as e:
            print(f"⚠️  翻译失败: {e},使用原文")
            return text

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

                        # 删除所有可能的集合 (TradingAgents 使用这三个)
                        collection_names = ['bull_memory', 'bear_memory', 'trader_memory']
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
        """执行真实的 TradingAgents 分析"""
        try:
            self.status = 'running'
            self.current_stage = '准备环境...'
            self.progress = 3

            # ========== 清理 ChromaDB (避免集合冲突) ==========
            self.cleanup_chromadb()

            self.current_stage = '导入 TradingAgents 模块...'
            self.progress = 5

            # 导入 TradingAgents
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.default_config import DEFAULT_CONFIG

            self.current_stage = '配置分析参数...'
            self.progress = 10

            # 创建配置
            config = DEFAULT_CONFIG.copy()

            # ⚠️ 优化配置以避免速率限制 (适合 Tier 1: 30K TPM)
            # 如果您的账户是 Tier 2+ (450K+ TPM),可以改为 'gpt-4o'
            config["deep_think_llm"] = self.config.get('deepThinkLLM', 'gpt-4o-mini')
            config["quick_think_llm"] = self.config.get('quickThinkLLM', 'gpt-4o-mini')

            # ⚠️ 不要在这里设置中文,保持分析用英文以获得更全面的信息
            # 只在最后翻译输出结果

            # 减少辩论轮次以降低 token 消耗
            config["max_debate_rounds"] = self.config.get('maxDebateRounds', 1)  # 从 2 改为 1

            # 根据研究深度调整配置 (进一步优化)
            depth = self.config.get('researchDepth', 'light')  # 默认改为 light
            if depth == 'light':
                config["max_debate_rounds"] = 1
            elif depth == 'deep':
                config["max_debate_rounds"] = 3

            self.current_stage = '初始化 TradingAgents...'
            self.progress = 15

            # 创建 TradingAgents 实例
            ta = TradingAgentsGraph(debug=True, config=config)

            # 获取股票代码和日期
            ticker = self.config['tickers'][0]  # 目前只支持第一个股票
            date = self.config['date']

            self.current_stage = f'开始分析 {ticker}...'
            self.progress = 20

            print("\n" + "🚀" * 30)
            print(f"开始 TradingAgents 分析: {ticker} @ {date}")
            print(f"配置: deep_think={config['deep_think_llm']}, quick_think={config['quick_think_llm']}")
            print(f"辩论轮数: {config.get('max_debate_rounds', 2)}")
            print("🚀" * 30 + "\n")

            # 运行分析 - 这是真实的 TradingAgents 调用!
            # propagate 方法会运行所有分析师、研究团队、交易员和风险管理
            import time
            start_time = time.time()

            print("⏱️  TradingAgents 分析开始...")
            state, decision = ta.propagate(ticker, date)

            elapsed_time = time.time() - start_time
            print(f"\n⏱️  TradingAgents 分析完成! 耗时: {elapsed_time:.1f} 秒")
            print("=" * 60)

            # 🔍 调试输出 - 帮助理解返回格式
            print("=" * 60)
            print("🔍 TradingAgents 返回数据调试:")
            print("=" * 60)
            print(f"State 类型: {type(state)}")
            print(f"Decision 类型: {type(decision)}")

            if isinstance(state, dict):
                print(f"\nState 包含的键: {list(state.keys())}")
                print(f"State 键的数量: {len(state.keys())}")

                # 详细打印每个键的类型和内容预览
                for key in sorted(state.keys()):
                    value = state[key]
                    value_type = type(value).__name__

                    # 获取内容预览
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                    elif isinstance(value, list):
                        preview = f"列表,长度: {len(value)}"
                    elif isinstance(value, dict):
                        preview = f"字典,键: {list(value.keys())[:3]}"
                    else:
                        preview = str(value)[:100]

                    print(f"  📌 {key}: [{value_type}] {preview}")

                # 特别关注分析师相关的字段
                print("\n🎯 分析师相关字段:")
                analyst_keys = [k for k in state.keys() if 'analyst' in k.lower() or
                                                           'analysis' in k.lower() or
                                                           'fundamental' in k.lower() or
                                                           'technical' in k.lower() or
                                                           'sentiment' in k.lower() or
                                                           'market' in k.lower() or
                                                           'news' in k.lower()]
                if analyst_keys:
                    for key in analyst_keys:
                        print(f"  ✅ {key}")
                else:
                    print("  ⚠️  未找到明显的分析师字段")
                    print("  📝 可能数据在 messages 字段中")

            else:
                print(f"\nState 内容 (前500字符): {str(state)[:500]}")

            if isinstance(decision, dict):
                print(f"\nDecision 包含的键: {list(decision.keys())}")
                print(f"Decision 内容: {decision}")
            else:
                print(f"\nDecision 内容 (前500字符): {str(decision)[:500]}")
            print("=" * 60)

            self.progress = 90
            self.current_stage = '解析分析结果...'

            # 将 TradingAgents 的输出转换为前端格式
            self.result = self.convert_to_frontend_format(state, decision, ticker, date)

            self.status = 'completed'
            self.current_stage = '分析完成'
            self.progress = 100

        except ImportError as e:
            self.status = 'failed'
            self.error = f'无法导入 TradingAgents: {str(e)}. 请确保 TradingAgents 已正确安装。'
            print(f"导入错误: {e}")
            traceback.print_exc()

        except Exception as e:
            self.status = 'failed'
            self.error = f'分析失败: {str(e)}'
            print(f"分析失败: {e}")
            traceback.print_exc()

    def convert_to_frontend_format(self, state, decision, ticker, date):
        """将 TradingAgents 的输出转换为前端需要的格式"""
        try:
            # 处理 decision - 可能是字符串或字典
            if isinstance(decision, str):
                # decision 是字符串,需要解析
                decision_text = decision

                # 翻译为中文
                if self.enable_translation and len(decision_text) > 50:
                    print(f"📝 原始分析长度: {len(decision_text)} 字符")
                    print(f"🌐 翻译启用状态: {self.enable_translation}")
                    decision_text = self.translate_to_chinese(decision_text)
                    print(f"📝 翻译后长度: {len(decision_text)} 字符")
                elif not self.enable_translation:
                    print(f"⚠️  翻译已禁用,使用英文原文")
                elif len(decision_text) <= 50:
                    print(f"⚠️  文本太短 ({len(decision_text)} 字符),跳过翻译")

                # 从文本中提取决策
                if 'BUY' in decision_text.upper() or '买入' in decision_text:
                    decision_action = 'BUY'
                elif 'SELL' in decision_text.upper() or '卖出' in decision_text:
                    decision_action = 'SELL'
                else:
                    decision_action = 'HOLD'

                # 使用整个文本作为建议
                recommendation = decision_text
                confidence = 0.7  # 默认置信度

            elif isinstance(decision, dict):
                # decision 是字典
                action = decision.get('action', 'HOLD')
                if action.upper() in ['BUY', 'SELL', 'HOLD']:
                    decision_action = action.upper()
                else:
                    decision_action = 'HOLD'

                recommendation = decision.get('reasoning', '') or decision.get('recommendation', '根据分析结果，请谨慎决策。')
                confidence = decision.get('confidence', 0.7)
            else:
                # 未知格式,使用默认值
                decision_action = 'HOLD'
                recommendation = '无法解析决策结果'
                confidence = 0.5

            # 提取价格信息 - 需要从多个可能的来源获取
            current_price = 0
            target_price = 0
            risks = []

            if isinstance(state, dict):
                # 尝试多种可能的字段名
                current_price = (
                    state.get('current_price') or
                    state.get('price') or
                    state.get('stock_price') or
                    0
                )

                target_price = (
                    state.get('target_price') or
                    state.get('price_target') or
                    0
                )

                risks = state.get('risks', [])

                # 如果还没有价格,尝试从 decision 中获取
                if isinstance(decision, dict) and current_price == 0:
                    current_price = decision.get('current_price', 0)

                if isinstance(decision, dict) and target_price == 0:
                    target_price = decision.get('target_price', 0)

            # 如果仍然没有价格,尝试实时获取
            if current_price == 0:
                try:
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('currentPrice', 0) or stock.info.get('regularMarketPrice', 0)
                    print(f"💰 从 yfinance 获取当前价格: ${current_price}")
                except Exception as e:
                    print(f"⚠️  无法获取当前价格: {e}")
                    current_price = 0

            # 如果有当前价格但没有目标价,使用当前价格
            if current_price > 0 and target_price == 0:
                # 检查 decision 文本中是否提到目标价
                if isinstance(decision, (str, dict)):
                    decision_str = str(decision)
                    # 尝试从文本中提取目标价 (简单正则)
                    import re
                    price_pattern = r'\$(\d+\.?\d*)'
                    matches = re.findall(price_pattern, decision_str)
                    if matches:
                        # 取最后一个提到的价格作为目标价
                        try:
                            target_price = float(matches[-1])
                            print(f"🎯 从决策文本中提取目标价: ${target_price}")
                        except:
                            pass

                # 如果还是没有,使用当前价格
                if target_price == 0:
                    target_price = current_price
                    print(f"⚠️  未找到目标价,使用当前价格: ${current_price}")

            # 计算上涨空间
            if current_price > 0 and target_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
            else:
                upside = 0

            # 提取分析师报告
            analysts = self.extract_analyst_reports(state)

            # 确保风险是列表
            if not risks:
                risks = ['暂无具体风险评估']

            return {
                'decision': decision_action,
                'confidence': float(confidence),
                'targetPrice': float(target_price) if target_price else 0,
                'currentPrice': float(current_price) if current_price else 0,
                'upside': float(upside),
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'date': date,
                'config': self.config,
                'analysts': analysts,
                'risks': risks if isinstance(risks, list) else [str(risks)],
                'recommendation': recommendation
            }

        except Exception as e:
            print(f"转换格式时出错: {e}")
            traceback.print_exc()
            # 返回基本格式
            return {
                'decision': 'HOLD',
                'confidence': 0.5,
                'targetPrice': 0,
                'currentPrice': 0,
                'upside': 0,
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'date': date,
                'config': self.config,
                'analysts': {},
                'risks': [f'解析错误: {str(e)}'],
                'recommendation': '分析过程中遇到问题，请检查日志。'
            }

    def extract_analyst_reports(self, state):
        """从 TradingAgents 状态中提取分析师报告 - 改进版"""
        analysts = {}

        try:
            print("\n" + "=" * 60)
            print("🔍 开始提取分析师报告")
            print("=" * 60)

            # 检查 state 是否是字典
            if not isinstance(state, dict):
                print(f"❌ State 不是字典,类型: {type(state)}")
                return self.get_default_analysts()

            print(f"✅ State 是字典,包含 {len(state)} 个键")
            print(f"   可用的键: {list(state.keys())}")

            # TradingAgents 通常将数据存储在 'messages' 字段中
            if 'messages' in state:
                print(f"\n📨 发现 messages 字段,包含 {len(state['messages'])} 条消息")
                analysts = self.extract_from_messages(state['messages'])

            # 尝试其他可能的字段
            for key in state.keys():
                if 'analyst' in key.lower():
                    print(f"📊 发现分析师字段: {key}")
                    # 跳过已经处理过的
                    analyst_key = key.replace('_analyst', '').replace('_analysis', '')
                    if analyst_key not in analysts:
                        analysts[analyst_key] = self.parse_analyst_data(
                            state[key],
                            key.replace('_', ' ').title(),
                            analyst_key
                        )

            # 检查是否有直接的分析字段
            # TradingAgents 实际使用的字段名!
            analyst_field_mapping = {
                'fundamental': ['fundamls_report', 'fundamental_report', 'fundamental_analysis', 'fundamentals_report'],  # 注意 fundamls 拼写
                'technical': ['market_report', 'technical_report', 'technical_analysis'],
                'sentiment': ['sentiment_report', 'sentiment_analysis'],
                'news': ['news_report', 'news_analysis']
            }

            for analyst_type, possible_fields in analyst_field_mapping.items():
                if analyst_type in analysts:
                    continue  # 已经有了

                for field_name in possible_fields:
                    if field_name in state and state[field_name]:
                        print(f"📊 发现 {analyst_type} 分析: {field_name}")
                        analysts[analyst_type] = self.parse_analyst_data(
                            self.translate_to_chinese(state[field_name]),
                            self.get_analyst_chinese_name(analyst_type),
                            analyst_type
                        )
                        break

            # 如果仍然没有找到任何分析数据
            if not analysts or len(analysts) < 2:
                print("⚠️  分析师数据不足,尝试从 messages 中深度解析...")
                if 'messages' in state:
                    extracted = self.extract_from_messages_advanced(state['messages'])
                    # 合并结果
                    for key, value in extracted.items():
                        if key not in analysts:
                            analysts[key] = value

            # 最后的兜底
            if not analysts:
                print("⚠️  所有尝试都失败,使用默认结构")
                analysts = self.get_default_analysts()
            else:
                print(f"✅ 成功提取 {len(analysts)} 个分析师的数据")
                # 确保至少有4个分析师
                for analyst_type in ['fundamental', 'technical', 'sentiment', 'news']:
                    if analyst_type not in analysts:
                        analysts[analyst_type] = self.get_default_analyst(
                            self.get_analyst_chinese_name(analyst_type)
                        )

        except Exception as e:
            print(f"❌ 提取分析师报告时出错: {e}")
            traceback.print_exc()
            analysts = self.get_default_analysts()

        print("=" * 60)
        return analysts

    def get_analyst_chinese_name(self, analyst_type):
        """获取分析师的中文名称"""
        names = {
            'fundamental': '基本面分析师',
            'technical': '技术分析师',
            'sentiment': '情绪分析师',
            'news': '新闻分析师'
        }
        return names.get(analyst_type, analyst_type)

    def extract_from_messages(self, messages):
        """从 LangGraph messages 中提取分析师报告"""
        analysts = {}

        try:
            for msg in messages:
                # LangGraph 消息通常有 'content' 和可能的 'name' 字段
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', '')
                else:
                    content = str(msg)

                if not content or len(content) < 20:
                    continue

                # 识别分析师类型
                content_lower = content.lower()

                if ('fundamental' in content_lower or '基本面' in content_lower) and 'fundamental' not in analysts:
                    analysts['fundamental'] = self.parse_text_to_analyst(
                        content, '基本面分析师', 'fundamental'
                    )

                if ('technical' in content_lower or '技术' in content_lower) and 'technical' not in analysts:
                    analysts['technical'] = self.parse_text_to_analyst(
                        content, '技术分析师', 'technical'
                    )

                if ('sentiment' in content_lower or '情绪' in content_lower or 'social' in content_lower) and 'sentiment' not in analysts:
                    analysts['sentiment'] = self.parse_text_to_analyst(
                        content, '情绪分析师', 'sentiment'
                    )

                if ('news' in content_lower or '新闻' in content_lower) and 'news' not in analysts:
                    analysts['news'] = self.parse_text_to_analyst(
                        content, '新闻分析师', 'news'
                    )

        except Exception as e:
            print(f"从 messages 提取时出错: {e}")
            traceback.print_exc()

        return analysts

    def extract_from_messages_advanced(self, messages):
        """高级消息提取 - 尝试从任何消息中提取有用信息"""
        analysts = {
            'fundamental': {'name': '基本面分析师', 'sentiment': '分析中', 'score': 5.0, 'summary': '', 'keyPoints': []},
            'technical': {'name': '技术分析师', 'sentiment': '分析中', 'score': 5.0, 'summary': '', 'keyPoints': []},
            'sentiment': {'name': '情绪分析师', 'sentiment': '分析中', 'score': 5.0, 'summary': '', 'keyPoints': []},
            'news': {'name': '新闻分析师', 'sentiment': '分析中', 'score': 5.0, 'summary': '', 'keyPoints': []},
        }

        try:
            all_content = []

            for msg in messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', '')
                else:
                    content = str(msg)

                if content and len(content) > 50:  # 只保留有实质内容的
                    all_content.append(content)

            print(f"   提取了 {len(all_content)} 条有效消息")

            # 将所有内容组合起来
            combined_content = "\n\n".join(all_content[:20])  # 最多取前20条

            # 至少显示一些内容
            if combined_content:
                # 简单分配给各个分析师
                summary_length = min(300, len(combined_content) // 4)

                for i, (key, analyst) in enumerate(analysts.items()):
                    start = i * summary_length
                    end = (i + 1) * summary_length
                    excerpt = combined_content[start:end] if start < len(combined_content) else "分析进行中..."

                    analyst['summary'] = excerpt[:250] if excerpt else "正在分析..."

                    # 提取关键点
                    sentences = [s.strip() + '。' for s in excerpt.split('.') if len(s.strip()) > 15]
                    analyst['keyPoints'] = sentences[:3] if sentences else ["分析进行中"]

                    print(f"   为 {analyst['name']} 分配了 {len(analyst['summary'])} 字符的摘要")

        except Exception as e:
            print(f"高级提取出错: {e}")
            traceback.print_exc()

        return analysts

    def parse_text_to_analyst(self, text, name, analyst_type):
        """将文本解析为分析师数据结构"""
        try:
            import re

            # 提取情绪
            sentiment = '中性'
            if any(word in text.lower() for word in ['positive', 'bullish', '看涨', '积极', '乐观', 'buy']):
                sentiment = '积极'
            elif any(word in text.lower() for word in ['negative', 'bearish', '看跌', '消极', '悲观', 'sell']):
                sentiment = '消极'

            # 尝试提取评分
            score = 6.0
            score_match = re.search(r'score[:\s]+([0-9.]+)', text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    score = 6.0

            # 提取摘要 (前250字符)
            summary = text[:250] + "..." if len(text) > 250 else text

            # 尝试提取关键点
            key_points = []

            # 方法1: 查找列表项
            bullet_points = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', text)
            if bullet_points:
                key_points = [bp.strip() for bp in bullet_points if len(bp.strip()) > 10][:5]

            # 方法2: 按句子分割
            if not key_points:
                sentences = [s.strip() + '。' for s in text.split('.') if len(s.strip()) > 20]
                key_points = sentences[:3]

            # 方法3: 至少返回一些内容
            if not key_points:
                key_points = [summary[:100]]

            return {
                'name': name,
                'sentiment': sentiment,
                'score': score,
                'summary': summary,
                'keyPoints': key_points[:5]  # 最多5个要点
            }

        except Exception as e:
            print(f"解析文本到分析师结构时出错: {e}")
            return self.get_default_analyst(name)

    def parse_analyst_data(self, data, name, analyst_type):
        """解析单个分析师的数据 - 改进版"""
        try:
            print(f"   解析 {name} 数据,类型: {type(data)}")

            if isinstance(data, dict):
                result = {
                    'name': name,
                    'sentiment': data.get('sentiment', data.get('tone', '中性')),
                    'score': float(data.get('score', data.get('rating', 5.0))),
                    'summary': data.get('summary', data.get('analysis', data.get('content', '暂无分析摘要'))),
                    'keyPoints': data.get('key_points', data.get('keyPoints', data.get('points', ['暂无要点'])))
                }
                print(f"   ✅ 成功解析字典数据")
                return result

            elif isinstance(data, str):
                # 如果是字符串,进行文本解析
                print(f"   📝 数据是字符串,长度: {len(data)}")
                return self.parse_text_to_analyst(data, name, analyst_type)

            elif isinstance(data, list):
                # 如果是列表,取第一个元素或组合
                print(f"   📋 数据是列表,长度: {len(data)}")
                if data:
                    return self.parse_analyst_data(data[0], name, analyst_type)
                else:
                    return self.get_default_analyst(name)

            else:
                print(f"   ⚠️  未知数据类型: {type(data)}")
                return self.get_default_analyst(name)

        except Exception as e:
            print(f"   ❌ 解析出错: {e}")
            traceback.print_exc()
            return self.get_default_analyst(name)

    def get_default_analyst(self, name):
        """获取默认的分析师结构"""
        return {
            'name': name,
            'sentiment': '分析中',
            'score': 5.0,
            'summary': '正在生成分析报告...',
            'keyPoints': ['分析正在进行中']
        }

    def get_default_analysts(self):
        """获取所有默认分析师"""
        return {
            'fundamental': self.get_default_analyst('基本面分析师'),
            'technical': self.get_default_analyst('技术分析师'),
            'sentiment': self.get_default_analyst('情绪分析师'),
            'news': self.get_default_analyst('新闻分析师')
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 尝试导入 TradingAgents 以验证安装
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        tradingagents_available = True
    except:
        tradingagents_available = False

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tradingagents_available': tradingagents_available,
        'version': 'real-tradingagents-v1.0'
    })

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """启动新的分析"""
    try:
        config = request.json

        # 验证必需字段
        if not config.get('tickers') or len(config['tickers']) == 0:
            return jsonify({'error': '至少需要一个股票代码'}), 400

        # 生成唯一分析 ID
        analysis_id = str(uuid.uuid4())

        # 创建并启动分析运行器
        runner = AnalysisRunner(analysis_id, config)
        analyses[analysis_id] = runner

        # 在后台线程中运行分析
        thread = threading.Thread(target=runner.run)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'analysisId': analysis_id,
            'message': '分析已启动'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/status', methods=['GET'])
def get_analysis_status(analysis_id):
    """获取正在进行的分析状态"""
    if analysis_id not in analyses:
        return jsonify({'error': '未找到分析'}), 404

    runner = analyses[analysis_id]

    return jsonify({
        'analysisId': analysis_id,
        'status': runner.status,
        'progress': runner.progress,
        'currentStage': runner.current_stage,
        'error': runner.error
    })

@app.route('/api/analysis/<analysis_id>/results', methods=['GET'])
def get_analysis_results(analysis_id):
    """获取已完成分析的结果"""
    if analysis_id not in analyses:
        return jsonify({'error': '未找到分析'}), 404

    runner = analyses[analysis_id]

    if runner.status != 'completed':
        return jsonify({
            'error': '分析未完成',
            'status': runner.status
        }), 400

    return jsonify({
        'analysisId': analysis_id,
        'result': runner.result
    })

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

@app.route('/api/config/models', methods=['GET'])
def get_available_models():
    """获取可用的 LLM 模型列表"""
    return jsonify({
        'deepThinkModels': [
            {'value': 'gpt-5.2', 'label': 'GPT-5.2 (最新)', 'description': '最先进的推理能力'},
            {'value': 'o1', 'label': 'o1 (推荐)', 'description': '深度思考模型'},
            {'value': 'o1-mini', 'label': 'o1-mini', 'description': '快速推理版本'},
            {'value': 'gpt-4o', 'label': 'GPT-4o', 'description': '多模态能力强'},
            {'value': 'claude-sonnet-4', 'label': 'Claude Sonnet 4', 'description': 'Anthropic 最新模型'}
        ],
        'quickThinkModels': [
            {'value': 'gpt-5.2-mini', 'label': 'GPT-5.2 Mini (推荐)', 'description': '最新轻量模型'},
            {'value': 'gpt-4o-mini', 'label': 'GPT-4o Mini', 'description': '高性价比'},
            {'value': 'gpt-4o', 'label': 'GPT-4o', 'description': '更高质量'},
            {'value': 'claude-haiku-4', 'label': 'Claude Haiku 4', 'description': '快速响应'}
        ],
        'researchDepths': [
            {'value': 'light', 'label': '轻度', 'description': '快速分析'},
            {'value': 'medium', 'label': '中度', 'description': '平衡方法'},
            {'value': 'deep', 'label': '深度', 'description': '全面分析'}
        ]
    })

if __name__ == '__main__':
    import os

    # 从环境变量读取配置 (支持 Railway 部署)
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    print("=" * 60)
    print("TradingAgents API Server")
    print("=" * 60)
    print(f"🚀 启动服务器在 {host}:{port}")
    print(f"   调试模式: {debug}")
    print(f"   环境: {'Development' if debug else 'Production'}")
    print("\n确保已设置环境变量:")
    print("  ✅ OPENAI_API_KEY")
    print("  ✅ ALPHA_VANTAGE_API_KEY")
    print("=" * 60)

    # 运行 Flask 应用
    app.run(host=host, port=port, debug=debug, threaded=True)
