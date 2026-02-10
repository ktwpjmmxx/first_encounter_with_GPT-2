import { useState } from 'react'
import { Plus, History, Settings, Search, Sparkles, AlertTriangle, CheckCircle, AlertCircle, Shield, Scale, FileText, Users, ChevronRight, Clock, ArrowRight, Zap, Activity, Eye } from 'lucide-react'

function App() {
  const [inputText, setInputText] = useState('')
  const [showResults, setShowResults] = useState(false)
  const [isLoading, setIsLoading] = useState(false) // ★ローディング状態追加
  const [activeNav, setActiveNav] = useState('new')
  
  // ★ APIからの結果を格納するステート
  const [assessmentResult, setAssessmentResult] = useState({
    riskScore: 0,
    riskLevel: 'Low',
    summary: '',
    laws: [],
    reason: '',
    recommendations: []
  })

  const handleInputChange = (e) => {
    setInputText(e.target.value)
  }

  const handleNavClick = (nav) => {
    setActiveNav(nav)
  }

  const handleDemoClick = (demo) => {
    setInputText(demo)
  }

  // ★ バックエンドへの問い合わせ処理
  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setShowResults(false);

    try {
      const response = await fetch('http://localhost:8000/api/assess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();

      // ステート更新
      setAssessmentResult({
        riskScore: data.risk_score,
        riskLevel: data.risk_level,
        summary: data.summary,
        laws: data.laws || [],
        reason: data.reason,
        recommendations: data.recommendations || []
      });

      setShowResults(true);

    } catch (error) {
      console.error("Error:", error);
      alert("診断中にエラーが発生しました。");
    } finally {
      setIsLoading(false);
    }
  }

  // アイコンのマッピング用ヘルパー
  const getIconForRec = (title) => {
    if (title.includes('privacy') || title.includes('data')) return Shield;
    if (title.includes('consent')) return Users;
    if (title.includes('retention')) return Clock;
    return FileText;
  };

  // リスクスコアによるゲージの計算
  const riskAngle = (assessmentResult.riskScore / 100) * 180;

  return (
    <div className="min-h-screen bg-[#0a0a0b] flex font-sans text-white">
      {/* Sidebar (Figrのデザインそのまま) */}
      <aside className="w-20 bg-[#0a0a0b] border-r border-white/5 flex flex-col items-center py-6">
        <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-2xl flex items-center justify-center mb-10 shadow-lg shadow-emerald-500/20">
          <Shield className="w-6 h-6 text-white" />
        </div>
        <nav className="flex-1 flex flex-col items-center gap-2">
          <button onClick={() => handleNavClick('new')} className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${activeNav === 'new' ? 'bg-white text-black' : 'text-white/40 hover:text-white hover:bg-white/5'}`}>
            <Plus className="w-5 h-5" />
          </button>
          <button onClick={() => handleNavClick('history')} className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${activeNav === 'history' ? 'bg-white text-black' : 'text-white/40 hover:text-white hover:bg-white/5'}`}>
            <History className="w-5 h-5" />
          </button>
        </nav>
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-400 to-pink-500 flex items-center justify-center text-sm font-bold cursor-pointer">
          JD
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-10">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Guardian AI</h1>
            <p className="text-white/30 text-sm">Legal Compliance Intelligence</p>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-6xl mx-auto p-10 space-y-8">
            
            {/* Input Section */}
            <div className="relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-500 rounded-3xl opacity-20 blur-xl" />
              <div className="relative bg-[#111113] rounded-3xl p-8 border border-white/10">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">New Assessment</h3>
                    <p className="text-white/40 text-sm">Describe your project for compliance analysis</p>
                  </div>
                </div>
                
                <textarea
                  value={inputText}
                  onChange={handleInputChange}
                  placeholder="Enter your project specifications..."
                  className="w-full h-32 bg-black/30 border border-white/10 rounded-2xl p-5 text-white placeholder-white/20 text-[15px] resize-none focus:outline-none focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all leading-relaxed"
                />
                
                <div className="flex items-center justify-between mt-6">
                  <div className="flex items-center gap-2">
                    <span className="text-white/30 text-sm">Examples:</span>
                    <button onClick={() => handleDemoClick('Points purchase system with cash withdrawal feature')} className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm rounded-xl transition-all">
                      Payment Flow
                    </button>
                    <button onClick={() => handleDemoClick('Collecting user face data for marketing')} className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm rounded-xl transition-all">
                      Privacy
                    </button>
                  </div>
                  
                  <button
                    onClick={handleAnalyze}
                    disabled={isLoading} // ★ロード中は無効化
                    className={`flex items-center gap-3 px-8 py-4 bg-white text-black font-semibold rounded-2xl hover:bg-white/90 transition-all ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isLoading ? 'Analyzing...' : 'Analyze'}
                    {!isLoading && <ArrowRight className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Results (条件付きレンダリング) */}
            {showResults && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
                
                {/* Risk Overview */}
                <div className="grid grid-cols-12 gap-6">
                  {/* Gauge Card */}
                  <div className="col-span-4 bg-[#111113] rounded-3xl p-8 border border-white/10 flex flex-col items-center justify-center">
                    <div className="relative w-48 h-48">
                      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" strokeLinecap="round" strokeDasharray="198 66" />
                        <circle cx="50" cy="50" r="42" fill="none" stroke="url(#gaugeGradient)" strokeWidth="8" strokeLinecap="round" strokeDasharray={`${riskAngle * 1.1} 264`} />
                        <defs>
                          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#10b981" />
                            <stop offset="50%" stopColor="#f59e0b" />
                            <stop offset="100%" stopColor="#ef4444" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-5xl font-bold">{assessmentResult.riskScore}</span>
                        <span className="text-white/40 text-sm">Risk Score</span>
                      </div>
                    </div>
                    
                    <div className={`mt-6 flex items-center gap-2 px-4 py-2 rounded-full ${
                      assessmentResult.riskLevel === 'High' ? 'bg-red-500/10 text-red-500' :
                      assessmentResult.riskLevel === 'Medium' ? 'bg-amber-500/10 text-amber-500' :
                      'bg-emerald-500/10 text-emerald-500'
                    }`}>
                      <AlertTriangle className="w-4 h-4" />
                      <span className="font-medium">{assessmentResult.riskLevel} Risk</span>
                    </div>
                  </div>

                  {/* Summary / Reason Card (Figrデザインを少し拡張して理由を表示) */}
                  <div className="col-span-8 bg-[#111113] rounded-3xl p-8 border border-white/10">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Activity className="w-5 h-5 text-cyan-500" />
                      AI Analysis
                    </h3>
                    <p className="text-white/70 leading-relaxed mb-6">
                      {assessmentResult.reason}
                    </p>
                    
                    <h4 className="text-sm font-semibold text-white/40 uppercase tracking-wider mb-3">Detected Laws</h4>
                    <div className="flex flex-wrap gap-3">
                      {assessmentResult.laws.map((tag, index) => (
                        <div key={index} className="flex items-center gap-3 px-5 py-3 bg-white/5 rounded-2xl">
                          <div className="w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400" />
                          <span className="font-medium">{tag.label}</span>
                          <span className="text-xs text-white/30 px-2 py-1 bg-white/5 rounded-lg">{tag.category}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-[#111113] rounded-3xl p-8 border border-white/10">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <Sparkles className="w-5 h-5 text-white/40" />
                      <h3 className="text-lg font-semibold">Action Items</h3>
                    </div>
                  </div>
                  <div className="space-y-3">
                    {assessmentResult.recommendations.map((rec, index) => {
                      const Icon = getIconForRec(rec.title.toLowerCase());
                      const isHigh = rec.priority.toLowerCase() === 'high';
                      
                      return (
                        <div key={index} className="flex items-center gap-5 p-5 bg-white/[0.02] border border-white/5 rounded-2xl">
                          <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 ${isHigh ? 'bg-red-500/10' : 'bg-amber-500/10'}`}>
                            <Icon className={`w-5 h-5 ${isHigh ? 'text-red-500' : 'text-amber-500'}`} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-3 mb-1">
                              <h4 className="font-medium">{rec.title}</h4>
                              <span className={`text-[10px] px-2 py-1 rounded-lg uppercase tracking-wider font-bold ${isHigh ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                                {rec.priority}
                              </span>
                            </div>
                            <p className="text-sm text-white/40 leading-relaxed">{rec.description}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App