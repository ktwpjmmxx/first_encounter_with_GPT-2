/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // 見出し用の明朝体設定
        serif: ['"Shippori Mincho B1"', 'Georgia', 'Cambria', '"Times New Roman"', 'Times', 'serif'],
        // 本文用のゴシック体設定
        sans: ['"Noto Sans JP"', 'sans-serif'],
      },
    },
  },
  plugins: [
    typography,
  ],
}