import { Clock, Star } from 'lucide-react';

const NewsCard = ({ article, onClick, rating, onRating }) => {
  // 評価用のスターコンポーネント（内部で使用）
  const StarRating = () => {
    const currentRating = rating || 0;
    
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={(e) => {
              e.stopPropagation(); // カード自体のクリックイベントを止める
              onRating(article.id, star);
            }}
            className="transition-transform hover:scale-110 focus:outline-none"
          >
            <Star
              className={`w-4 h-4 transition-colors ${
                star <= currentRating 
                  ? 'fill-[#C4A77D] text-[#C4A77D]' 
                  : 'text-[#D4CFC5]'
              }`}
            />
          </button>
        ))}
      </div>
    );
  };

  return (
    <article 
      onClick={onClick}
      className="bg-white rounded-xl border border-[#E8E4DD] overflow-hidden hover:shadow-lg hover:border-[#C4A77D] transition-all cursor-pointer group"
    >
      <div className="flex flex-col md:flex-row">
        <div className="w-full md:w-72 h-48 flex-shrink-0 overflow-hidden bg-gray-100">
          <img 
            src={article.image} 
            alt={article.title}
            referrerPolicy="no-referrer"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.target.src = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=1000";
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
          <h3 className="text-xl font-bold text-[#1A1A1A] mb-2 leading-tight group-hover:text-[#8B6914] transition-colors font-serif">
            {article.title}
          </h3>
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
              <StarRating />
              <span>{article.date}</span>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
};

export default NewsCard;