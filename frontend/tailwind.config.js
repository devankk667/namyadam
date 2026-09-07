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
        // Domain classification colors (semantic to detection taxonomy, not decorative)
        industrial: {
          fire: '#f85149',
          source: '#e3934a',
          wildfire: '#d29922',
          agri: '#3fb950',
          other: '#6b7684'
        },
        // Technical console surface system
        base: '#090c10',
        panel: '#0d1117',
        panel2: '#11161d',
        line: '#1e2530',
        line2: '#2a3341',
        ink: {
          primary: '#e6e9ef',
          secondary: '#9da8b7',
          muted: '#5f6b7a',
        },
        accent: {
          DEFAULT: '#e3934a',
          dim: '#e3934a1a',
          strong: '#f0a563',
        },
        // Status semantics — the ONLY colors that should carry meaning
        status: {
          success: '#3fb950',
          warning: '#d29922',
          critical: '#f85149',
          info: '#58a6ff',
          neutral: '#5f6b7a',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        none: 'none',
        overlay: '0 4px 16px 0 rgba(0,0,0,0.45)',
      },
    },
  },
  plugins: [],
}
