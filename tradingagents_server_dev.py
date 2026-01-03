"""
TradingAgents Frontend API Server
This Flask server acts as a bridge between the web frontend and the TradingAgents CLI
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import json
import os
import threading
import time
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Store for ongoing analyses
analyses = {}

class AnalysisRunner:
    """Manages running TradingAgents analysis in the background"""
    
    def __init__(self, analysis_id, config):
        self.analysis_id = analysis_id
        self.config = config
        self.status = 'initializing'
        self.progress = 0
        self.current_stage = '准备分析...'
        self.result = None
        self.error = None
        
    def run(self):
        """Execute the TradingAgents analysis"""
        try:
            self.status = 'running'
            self.current_stage = '启动 TradingAgents...'
            self.progress = 10
            
            # Build command for TradingAgents CLI
            # This would need to be adapted based on how TradingAgents CLI accepts parameters
            cmd = [
                'python', '-m', 'cli.main',
                '--tickers', ','.join(self.config.get('tickers', ['NVDA'])),
                '--date', self.config.get('date', datetime.now().strftime('%Y-%m-%d')),
                '--deep-llm', self.config.get('deepThinkLLM', 'gpt-5.2'),
                '--quick-llm', self.config.get('quickThinkLLM', 'gpt-5.2-mini'),
                '--depth', self.config.get('researchDepth', 'medium'),
                '--debate-rounds', str(self.config.get('maxDebateRounds', 2)),
                '--output-json'  # Assuming CLI can output JSON
            ]

            # Simulate progress updates
            stages = [
                ('初始化分析环境...', 10),
                ('基本面分析师工作中...', 20),
                ('技术分析师评估中...', 35),
                ('情绪分析师分析中...', 50),
                ('新闻分析师处理中...', 65),
                ('研究团队辩论中...', 75),
                ('交易员决策中...', 85),
                ('风险管理评估中...', 95),
                ('生成最终报告...', 100)
            ]
            
            for stage, progress in stages:
                self.current_stage = stage
                self.progress = progress
                time.sleep(2)  # Simulate work being done
            
            # In a real implementation, you would:
            # 1. Run the actual TradingAgents process
            # 2. Parse its output
            # 3. Store the results
            
            # For now, return mock data (中文版本)
            self.result = {
                'decision': 'BUY',
                'confidence': 0.78,
                'targetPrice': 145.50,
                'currentPrice': 128.30,
                'upside': 13.4,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'analysts': {
                    'fundamental': {
                        'name': '基本面分析师',
                        'sentiment': '积极',
                        'score': 8.5,
                        'summary': '公司财务表现强劲,营收增长稳健,基本面良好。',
                        'keyPoints': [
                            'Q3营收同比增长23%,显示强劲增长势头',
                            '毛利率提升至42%,盈利能力持续改善',
                            '现金储备充裕,负债率低,财务健康'
                        ]
                    },
                    'technical': {
                        'name': '技术分析师',
                        'sentiment': '看涨',
                        'score': 7.8,
                        'summary': 'MACD指标显示买入信号,RSI处于健康区间,技术面向好。',
                        'keyPoints': [
                            'MACD形成金叉,短期趋势转强',
                            'RSI指标为58,处于中性偏多区域',
                            '成功突破$125关键阻力位,上涨动能增强'
                        ]
                    },
                    'sentiment': {
                        'name': '情绪分析师',
                        'sentiment': '乐观',
                        'score': 8.2,
                        'summary': '社交媒体情绪积极,机构投资者持续增持,市场情绪向好。',
                        'keyPoints': [
                            '社交媒体提及量增加45%,关注度持续上升',
                            '散户情绪指数72/100,整体偏乐观',
                            '机构持仓环比增加8%,显示专业投资者信心'
                        ]
                    },
                    'news': {
                        'name': '新闻分析师',
                        'sentiment': '正面',
                        'score': 7.5,
                        'summary': '近期新闻整体偏正面,新产品发布获得市场积极反馈。',
                        'keyPoints': [
                            '新产品预订量超出预期,市场需求旺盛',
                            '与重要客户签订长期合作协议,订单有保障',
                            '多家分析师上调目标价,市场预期改善'
                        ]
                    }
                },
                'risks': [
                    '整体市场波动性仍然较高,需警惕系统性风险',
                    '行业竞争日益激烈,可能影响市场份额和利润率',
                    '宏观经济存在不确定性,可能影响消费需求'
                ],
                'recommendation': '综合多维度分析,建议适当建仓该股票。建议采用分批买入策略,初始仓位控制在总资产的5%以内,后续可根据市场表现和基本面变化调整仓位。设置止损位以控制风险,长期持有以获取价值增长。'
            }
            
            self.status = 'completed'
            self.current_stage = '分析完成'
            self.progress = 100
            
        except Exception as e:
            self.status = 'failed'
            self.error = str(e)
            print(f"分析失败: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Start a new analysis"""
    try:
        config = request.json
        
        # Validate required fields
        if not config.get('tickers') or len(config['tickers']) == 0:
            return jsonify({'error': 'At least one ticker is required'}), 400
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Create and start analysis runner
        runner = AnalysisRunner(analysis_id, config)
        analyses[analysis_id] = runner
        
        # Run analysis in background thread
        thread = threading.Thread(target=runner.run)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'analysisId': analysis_id,
            'message': 'Analysis started successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/status', methods=['GET'])
def get_analysis_status(analysis_id):
    """Get the status of an ongoing analysis"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
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
    """Get the results of a completed analysis"""
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

@app.route('/api/analysis', methods=['GET'])
def list_analyses():
    """List all analyses"""
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
    """Get list of available LLM models"""
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

# Serve the frontend
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file"""
    return send_file('tradingagents-frontend.html')

if __name__ == '__main__':
    print("=" * 60)
    print("TradingAgents Frontend API Server")
    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print("Make sure TradingAgents is installed and configured")
    print("=" * 60)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
