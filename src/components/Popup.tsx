import React from 'react';

interface PopupProps {
  isOpen: boolean;
  onClose: () => void;
  message: string;
}

const Popup: React.FC<PopupProps> = ({ isOpen, onClose, message }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50">
      <div className="absolute inset-0 bg-black bg-opacity-50" onClick={onClose}></div>
      <div className="relative bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg max-w-sm w-full">
        <button 
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          onClick={onClose}
        >
          ✕
        </button>
        <div className="text-center">
          <h2 className="text-xl font-bold mb-4">{message}</h2>
          <button
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            onClick={onClose}
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};

export default Popup;
