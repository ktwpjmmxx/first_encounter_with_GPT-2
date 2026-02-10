import { useState } from 'react'
import { TrendingUp, ChevronLeft, Bookmark, ExternalLink, Clock, Star } from 'lucide-react'

// 子コンポーネントのインポート
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import NewsCard from './components/NewsCard'

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
  const handleRemoveTopic = (topic) => setUserTopics(userTopics.filter(t => t !== topic));

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

  // --- 記事詳細画面 (まだApp.jsx内に残している場合) ---
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
          {/* ... 詳細画面のレイアウト (長くなるため中略、NewsCardと同様に切り出し推奨) ... */}
           {/* タイトル */}
           <h1 className="text-4xl font-bold text-[#1A1A1A] leading-tight mb-6 font-serif">
            {selectedArticle.title}
          </h1>
          {/* ... 画像や本文など既存のJSX ... */}
          <div className="mb-10 rounded-xl overflow-hidden bg-gray-100">
            <img src={selectedArticle.image} alt={selectedArticle.title} className="w-full h-80 object-cover" />
          </div>
          <div className="prose prose-lg max-w-none prose-headings:font-serif">
             <div dangerouslySetInnerHTML={{ __html: selectedArticle.content }} />
          </div>
          
          {/* 評価部分（再利用のためにコンポーネント化したいが、ここでは直書きのままの例） */}
          <div className="mt-12 pt-8 border-t border-[#E8E4DD]">
             {/* ... */}
          </div>
        </article>
      </div>
    )
  }

  // --- メイン画面 (HOME) ---
  return (
    <div className="min-h-screen bg-[#FAF8F5] font-sans">
      
      {/* 1. サイドバー（設定） */}
      <Sidebar 
        isOpen={showPreferences}
        onClose={() => setShowPreferences(false)}
        userTopics={userTopics}
        onRemoveTopic={handleRemoveTopic}
        onAddTopic={handleAddTopic}
      />

      {/* 2. ヘッダー（画像 + 検索 + トピック） */}
      <Header 
        onToggleSettings={() => setShowPreferences(!showPreferences)}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onSearch={() => fetchNews(activeTopic)}
        isLoading={isLoading}
        topics={topics}
        activeTopic={activeTopic}
        setActiveTopic={setActiveTopic}
      />

      {/* 3. メインコンテンツ */}
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
            <NewsCard 
              key={article.id}
              article={article}
              onClick={() => handleArticleClick(article)}
              rating={ratings[article.id]}
              onRating={handleRating}
            />
          ))}
        </div>
      </main>
    </div>
  )
}

export default App