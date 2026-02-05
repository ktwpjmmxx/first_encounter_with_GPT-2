import { useState } from 'react'
import { Search, X, Plus, Bookmark, ExternalLink, Clock, ChevronLeft, Settings, TrendingUp, Sparkles, Star } from 'lucide-react'

// ★設定: APIのURL
const API_BASE_URL = "http://127.0.0.1:8000"; 
const NEWS_URL = `${API_BASE_URL}/api/news`;
const FEEDBACK_URL = `${API_BASE_URL}/api/feedback`;

function App() {
  const [activeScreen, setActiveScreen] = useState('home')
  const [activeTopic, setActiveTopic] = useState('All')
  const [showPreferences, setShowPreferences] = useState(false)
  const [selectedArticle, setSelectedArticle] = useState(null)
  
  // ユーザー設定
  const [userTopics, setUserTopics] = useState(['生成AI', 'Apple', 'マーケティング', 'コーヒー', 'Space'])
  const [ratings, setRatings] = useState({})
  
  // 検索・記事データ
  const [searchQuery, setSearchQuery] = useState('') 
  const [articles, setArticles] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const topics = ['All', 'For You', ...userTopics]

  // --- バックエンド連携機能 ---
  const fetchNews = async (topic) => {
    setIsLoading(true);
    const queryTopic = topic || activeTopic;
    try {
      const response = await fetch(NEWS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: queryTopic, user_topics: userTopics }),
      });
      if (!response.ok) throw new Error('Network response');
      const data = await response.json();
      
      const formattedArticles = data.articles.map((art, index) => ({
        id: index,
        title: art.title,
        source: parseSource(art.url),
        image: art.image_url || 'https://images.unsplash.com/photo-1504711434969-e33886168f5c',
        tags: [art.search_source, art.source_badge || 'NEWS'],
        readTime: art.read_time,
        excerpt: art.editorial,
        content: art.full_story, 
        date: new Date().toLocaleDateString(),
        url: art.url
      }));
      setArticles(formattedArticles);
      if (topic === searchQuery) setSearchQuery('');
    } catch (error) {
      alert("ニュースの取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  const parseSource = (url) => {
    try { return new URL(url).hostname.replace('www.', ''); } catch { return 'News Source'; }
  };

  const handleRating = async (articleId, rating) => {
    setRatings(prev => ({ ...prev, [articleId]: rating }))
    const targetArticle = articles.find(a => a.id === articleId) || selectedArticle;
    if (!targetArticle) return;
    try {
      await fetch(FEEDBACK_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          article_title: targetArticle.title,
          rating: rating,
          tags: targetArticle.tags
        }),
      });
    } catch (error) { console.error(error); }
  }

  // --- UI操作 ---
  const handleRemoveTopic = (topic) => {
    setUserTopics(userTopics.filter(t => t !== topic))
  }

  const handleAddTopic = () => {
    const newTopic = prompt('Enter a new topic:')
    if (newTopic && !userTopics.includes(newTopic)) {
      setUserTopics([...userTopics, newTopic])
    }
  }

  const handleArticleClick = (article) => {
    setSelectedArticle(article)
    setActiveScreen('article')
  }

  const handleBackToFeed = () => {
    setSelectedArticle(null)
    setActiveScreen('home')
  }

  const togglePreferences = () => {
    setShowPreferences(!showPreferences)
  }

  // --- コンポーネント ---
  const StarRating = ({ articleId, size = 'md' }) => {
    const currentRating = ratings[articleId] || 0
    const starSize = size === 'sm' ? 'w-4 h-4' : 'w-5 h-5'
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={(e) => { e.stopPropagation(); handleRating(articleId, star); }}
            className="transition-transform hover:scale-110"
          >
            <Star className={`${starSize} ${star <= currentRating ? 'fill-[#C4A77D] text-[#C4A77D]' : 'text-[#D4CFC5]'} transition-colors`} />
          </button>
        ))}
      </div>
    )
  }

  // --- メイン画面 (HOME) ---
  if (activeScreen === 'home') {
    return (
      <div className="min-h-screen bg-[#FAF8F5] font-sans">
        {/* 設定サイドバー */}
        {showPreferences && (
          <div className="fixed inset-0 z-50 flex">
            <div className="absolute inset-0 bg-black/20" onClick={togglePreferences}></div>
            <div className="relative w-80 bg-[#FAF8F5] h-full shadow-2xl overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-lg font-semibold text-[#1A1A1A] tracking-wide uppercase font-serif">設定</h2>
                  <button onClick={togglePreferences} className="p-2 hover:bg-[#EDE8E0] rounded-full transition-colors">
                    <X className="w-5 h-5 text-[#666]" />
                  </button>
                </div>
                <div className="mb-8">
                  <h3 className="text-xs font-semibold text-[#999] uppercase tracking-wider mb-4">マイトピック</h3>
                  <div className="space-y-3">
                    {userTopics.map((topic, index) => (
                      <div key={index} className="flex items-center justify-between p-4 bg-white rounded-lg border border-[#E8E4DD] hover:border-[#C4A77D] transition-colors group">
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-[#C4A77D]"></div>
                          <span className="text-[#1A1A1A] font-medium">{topic}</span>
                        </div>
                        <button onClick={() => handleRemoveTopic(topic)} className="p-1 opacity-0 group-hover:opacity-100 hover:bg-[#F5F2ED] rounded transition-all">
                          <X className="w-4 h-4 text-[#999]" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
                <button onClick={handleAddTopic} className="w-full flex items-center justify-center gap-2 p-4 border-2 border-dashed border-[#D4CFC5] rounded-lg text-[#666] hover:border-[#C4A77D] hover:text-[#1A1A1A] transition-colors">
                  <Plus className="w-5 h-5" />
                  <span className="font-medium">トピックを追加</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ヘッダー */}
        <header className="relative">
          <div className="h-48 bg-cover bg-center relative" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=1600&q=80)' }}>
            <div className="absolute inset-0 bg-black/40"></div>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
              <h1 className="text-5xl font-serif tracking-wider" style={{ fontFamily: 'Playfair Display, Georgia, serif' }}>
                TOKYO STAPLE
              </h1>
              <div className="flex items-center gap-4 mt-3 text-sm tracking-widest opacity-90 font-serif">
                <span>VOL. CLXXIV</span>
                <span>•</span>
                <span>{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase()}</span>
                <span>•</span>
                <span>TOKYO EDITION</span>
              </div>
            </div>
          </div>

          <div className="bg-white border-b border-[#E8E4DD]">
            <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
              <button onClick={togglePreferences} className="flex items-center gap-2 px-4 py-2 text-sm text-[#666] hover:text-[#1A1A1A] hover:bg-[#F5F2ED] rounded-lg transition-colors">
                <Settings className="w-4 h-4" />
                <span>トピック</span>
              </button>
              
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#999]" />
                  <input 
                    type="text" 
                    placeholder="記事を検索..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && fetchNews(searchQuery)}
                    className="pl-10 pr-4 py-2 w-64 bg-[#F5F2ED] border-none rounded-lg text-sm placeholder:text-[#999] focus:outline-none focus:ring-2 focus:ring-[#C4A77D]"
                  />
                </div>
                <button 
                  onClick={() => fetchNews(activeTopic)}
                  disabled={isLoading}
                  className="bg-[#1A1A1A] text-white px-4 py-2 rounded-lg text-sm flex gap-2 items-center hover:bg-[#333] disabled:opacity-50"
                >
                  {isLoading ? "Generating..." : <><Sparkles className="w-4 h-4"/> CREATE</>}
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* トピックフィルター */}
        <div className="bg-white border-b border-[#E8E4DD]">
          <div className="max-w-5xl mx-auto px-6 py-3">
            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              {topics.map((topic) => (
                <button
                  key={topic}
                  onClick={() => setActiveTopic(topic)}
                  className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
                    activeTopic === topic ? 'bg-[#1A1A1A] text-white' : 'bg-[#F5F2ED] text-[#666] hover:bg-[#E8E4DD] hover:text-[#1A1A1A]'
                  }`}
                >
                  {topic === 'For You' && <Sparkles className="w-3 h-3 inline mr-1" />}
                  {topic}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* メインコンテンツ */}
        <main className="max-w-5xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-sm font-semibold text-[#999] uppercase tracking-wider">
              {activeTopic === 'All' ? '本日のエディション' : activeTopic}
            </h2>
            <div className="flex items-center gap-1 text-sm text-[#999]">
              <TrendingUp className="w-4 h-4" />
              <span>{articles.length} Stories Generated</span>
            </div>
          </div>

          <div className="space-y-6">
            {articles.length === 0 && !isLoading && (
              <div className="text-center py-20 text-[#999]">上の「CREATE」ボタンを押してニュースを生成してください。</div>
            )}
            
            {articles.map((article) => (
              <article 
                key={article.id} 
                onClick={() => handleArticleClick(article)}
                className="bg-white rounded-xl border border-[#E8E4DD] overflow-hidden hover:shadow-lg hover:border-[#C4A77D] transition-all cursor-pointer group"
              >
               <div className="flex flex-col md:flex-row">
                  <div className="w-full md:w-72 h-48 flex-shrink-0 overflow-hidden bg-gray-100">
                    {/* ★修正: 画像表示エラー対策 */}
                    <img 
                      src={article.image} 
                      alt={article.title}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      onError={(e) => {
                        e.target.src = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=1000"
                      }}
                    />
                  </div>
                  <div className="flex-1 p-6">
                    <div className="flex items-center gap-2 mb-3">
                      {article.tags.map((tag, i) => (
                        <span key={i} className="px-3 py-1 bg-[#F5F2ED] text-xs font-medium text-[#666] rounded-full">
                          {tag}
                        </span>
                      ))}
                    </div>
                    {/* タイトル (Shippori Mincho適用) */}
                    <h3 className="text-xl font-bold text-[#1A1A1A] mb-2 leading-tight group-hover:text-[#8B6914] transition-colors font-serif">
                      {article.title}
                    </h3>
                    {/* 概要 (Noto Sans JP適用) */}
                    <p className="text-[#666] text-sm leading-relaxed mb-4 line-clamp-2 font-sans">
                      {article.excerpt}
                    </p>
                    <div className="flex items-center justify-between text-xs text-[#999]">
                      <div className="flex items-center gap-4">
                        <span>{article.source}</span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {article.readTime}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <StarRating articleId={article.id} size="sm" />
                        <span>{article.date}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </main>
      </div>
    )
  }

  // --- 記事詳細画面 (ARTICLE DETAIL) ---
  if (activeScreen === 'article' && selectedArticle) {
    return (
      <div className="min-h-screen bg-[#FAF8F5]">
        <header className="bg-white border-b border-[#E8E4DD] sticky top-0 z-10">
          <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
            <button onClick={handleBackToFeed} className="flex items-center gap-2 text-[#666] hover:text-[#1A1A1A] transition-colors">
              <ChevronLeft className="w-5 h-5" />
              <span className="text-sm font-medium">フィードに戻る</span>
            </button>
            <div className="flex items-center gap-3">
              <button className="p-2 hover:bg-[#F5F2ED] rounded-lg transition-colors">
                <Bookmark className="w-5 h-5 text-[#666]" />
              </button>
              <a href={selectedArticle.url} target="_blank" className="flex items-center gap-2 px-4 py-2 bg-[#1A1A1A] text-white text-sm font-medium rounded-lg hover:bg-[#333] transition-colors">
                <ExternalLink className="w-4 h-4" />
                元記事
              </a>
            </div>
          </div>
        </header>

        <article className="max-w-3xl mx-auto px-6 py-12">
          {/* Meta */}
          <div className="flex items-center gap-2 mb-6">
            {selectedArticle.tags.map((tag, i) => (
              <span key={i} className="px-3 py-1 bg-[#E8E4DD] text-xs font-semibold text-[#666] rounded-full uppercase tracking-wide">
                {tag}
              </span>
            ))}
          </div>

          {/* Title (特大明朝体) */}
          <h1 className="text-4xl font-bold text-[#1A1A1A] leading-tight mb-6 font-serif">
            {selectedArticle.title}
          </h1>

          {/* Source & Date */}
          <div className="flex items-center gap-4 pb-8 border-b border-[#E8E4DD] mb-8">
            <span className="text-[#666] font-medium">{selectedArticle.source}</span>
            <span className="text-[#999]">•</span>
            <span className="text-[#999]">{selectedArticle.date}</span>
            <span className="text-[#999]">•</span>
            <span className="flex items-center gap-1 text-[#999]">
              <Clock className="w-4 h-4" />
              {selectedArticle.readTime}
            </span>
          </div>

          {/* Featured Image */}
          <div className="mb-10 rounded-xl overflow-hidden bg-gray-100">
            {/* ★修正: 画像表示エラー対策 */}
            <img 
              src={selectedArticle.image} 
              alt={selectedArticle.title} 
              referrerPolicy="no-referrer"
              className="w-full h-80 object-cover" 
              onError={(e) => {
                e.target.src = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=1000"
              }}
            />
          </div>

          {/* 本文のデザイン (見出し等のスタイルを強化) */}
          <div className="prose prose-lg max-w-none 
            prose-headings:font-serif prose-headings:font-bold prose-headings:text-[#1A1A1A] 
            
            /* H3見出しのデザイン */
            prose-h3:text-2xl 
            prose-h3:text-[#0F172A]
            prose-h3:mt-14 
            prose-h3:mb-6 
            prose-h3:border-l-4 
            prose-h3:border-[#C4A77D] 
            prose-h3:pl-4 
            
            /* 本文のデザイン */
            prose-p:font-sans
            prose-p:text-[#333] 
            prose-p:text-[17px]
            prose-p:leading-[2.0]
            prose-p:tracking-wide
            prose-p:mb-8 
            
            prose-li:font-sans
            prose-li:text-[#333] 
            prose-li:leading-loose">
             <div dangerouslySetInnerHTML={{ __html: selectedArticle.content }} />
          </div>

          {/* Feedback Section */}
          <div className="mt-12 pt-8 border-t border-[#E8E4DD]">
            <div className="bg-[#F5F2ED] rounded-xl p-6 mb-8">
              <h3 className="text-sm font-semibold text-[#1A1A1A] mb-2 font-serif">この記事を評価してください</h3>
              <p className="text-xs text-[#666] mb-4">あなたの評価はAIがより良いニュースを選ぶ参考になります</p>
              <div className="flex items-center gap-4">
                <StarRating articleId={selectedArticle.id} size="md" />
                {ratings[selectedArticle.id] && (
                  <span className="text-sm text-[#C4A77D] font-medium">
                    {ratings[selectedArticle.id]}点 - ありがとうございます！
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Related Topics */}
          <div className="pt-4">
            <h3 className="text-xs font-semibold text-[#999] uppercase tracking-wider mb-4">関連トピック</h3>
            <div className="flex flex-wrap gap-2">
              {selectedArticle.tags.map((tag, i) => (
                <button key={i} className="px-4 py-2 bg-white border border-[#E8E4DD] rounded-full text-sm text-[#666] hover:border-[#C4A77D] hover:text-[#1A1A1A] transition-colors">
                  {tag}
                </button>
              ))}
              <button className="px-4 py-2 bg-white border border-[#E8E4DD] rounded-full text-sm text-[#666] hover:border-[#C4A77D] hover:text-[#1A1A1A] transition-colors">
                リーガルテック
              </button>
              <button className="px-4 py-2 bg-white border border-[#E8E4DD] rounded-full text-sm text-[#666] hover:border-[#C4A77D] hover:text-[#1A1A1A] transition-colors">
                ビジネス戦略
              </button>
            </div>
          </div>
        </article>
      </div>
    )
  }

  return null
}

export default App