import { X, Plus } from 'lucide-react';

const Sidebar = ({ isOpen, onClose, userTopics, onRemoveTopic, onAddTopic }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* 背景クリックで閉じる */}
      <div className="absolute inset-0 bg-black/20" onClick={onClose}></div>
      
      <div className="relative w-80 bg-[#FAF8F5] h-full shadow-2xl overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-lg font-semibold text-[#1A1A1A] tracking-wide uppercase font-serif">設定</h2>
            <button onClick={onClose} className="p-2 hover:bg-[#EDE8E0] rounded-full transition-colors">
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
                  <button onClick={() => onRemoveTopic(topic)} className="p-1 opacity-0 group-hover:opacity-100 hover:bg-[#F5F2ED] rounded transition-all">
                    <X className="w-4 h-4 text-[#999]" />
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          <button onClick={onAddTopic} className="w-full flex items-center justify-center gap-2 p-4 border-2 border-dashed border-[#D4CFC5] rounded-lg text-[#666] hover:border-[#C4A77D] hover:text-[#1A1A1A] transition-colors">
            <Plus className="w-5 h-5" />
            <span className="font-medium">トピックを追加</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;