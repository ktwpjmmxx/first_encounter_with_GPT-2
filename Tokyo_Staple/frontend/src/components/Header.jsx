import { Search, Settings, Sparkles } from 'lucide-react';

const Header = ({ 
  onToggleSettings, 
  searchQuery, 
  setSearchQuery, 
  onSearch, 
  isLoading, 
  topics, 
  activeTopic, 
  setActiveTopic 
}) => {
  return (
    <>
      <header className="relative">
        {/* ヒーロー画像エリア */}
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

        {/* 検索バーエリア */}
        <div className="bg-white border-b border-[#E8E4DD]">
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
            <button onClick={onToggleSettings} className="flex items-center gap-2 px-4 py-2 text-sm text-[#666] hover:text-[#1A1A1A] hover:bg-[#F5F2ED] rounded-lg transition-colors">
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
                  onKeyDown={(e) => e.key === 'Enter' && onSearch(searchQuery)}
                  className="pl-10 pr-4 py-2 w-64 bg-[#F5F2ED] border-none rounded-lg text-sm placeholder:text-[#999] focus:outline-none focus:ring-2 focus:ring-[#C4A77D]"
                />
              </div>
              <button 
                onClick={() => onSearch()}
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
    </>
  );
};

export default Header;