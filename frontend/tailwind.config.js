/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          850: '#1e293b',
          950: '#020617',
        },
        industrial: {
          fire: '#ef4444',
          source: '#f97316',
          wildfire: '#eab308',
          agri: '#10b981',
          other: '#6b7280'
        }
      }
    },
  },
  plugins: [],
}
