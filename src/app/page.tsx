"use client";

import { useState } from "react";
import Popup from "@/components/Popup";

export default function Home() {
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const openPopup = () => {
    setIsPopupOpen(true);
  };

  const closePopup = () => {
    setIsPopupOpen(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
      <main className="flex flex-col items-center justify-center w-full flex-1 px-4 sm:px-20 text-center">
        <h1 className="text-4xl font-bold mt-10 mb-8 text-gray-900 dark:text-white">
          Hello World アプリ
        </h1>
        
        <button
          onClick={openPopup}
          className="px-6 py-3 bg-blue-500 text-white font-medium rounded-lg shadow-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
        >
          ポップアップを表示
        </button>
        
        <Popup 
          isOpen={isPopupOpen} 
          onClose={closePopup} 
          message="Hello World!"
        />
      </main>
      
      <footer className="w-full h-16 border-t border-gray-200 dark:border-gray-800 flex items-center justify-center">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Next.js + TypeScript + Tailwind CSS サンプル
        </p>
      </footer>
    </div>
  );
}
