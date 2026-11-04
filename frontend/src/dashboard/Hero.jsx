// Decorative artwork only. Nothing here depends on data: the splashes, waves, rain and droplets are an illustration, hidden
// from assistive technology, and they never show or imply a live flood condition.
const SPLASH_A =
  "M72.1 -2.0 C71.7 -0.4 28.7 0.0 31.5 5.0 C34.3 9.9 89.6 26.1 88.8 27.6 C87.9 29.1 29.1 9.0 26.3 13.9 C23.5 18.7 73.1 55.5 71.7 56.6 C70.3 57.8 24.8 19.9 17.9 20.6 C11.1 21.2 32.5 59.3 30.9 60.3 C29.2 61.3 11.1 18.6 7.9 26.5 C4.6 34.5 13.0 107.8 11.3 108.1 C9.6 108.3 3.4 30.0 -2.2 27.8 C-7.9 25.7 -20.7 94.7 -22.6 95.2 C-24.5 95.8 -8.9 35.8 -13.8 31.2 C-18.8 26.6 -50.3 68.9 -52.3 67.5 C-54.3 66.1 -19.5 23.3 -25.6 22.8 C-31.8 22.3 -89.3 66.5 -89.4 64.7 C-89.5 63.0 -29.9 20.5 -26.4 12.5 C-22.9 4.5 -67.8 18.6 -68.3 16.8 C-68.8 15.0 -28.7 7.3 -29.3 1.7 C-29.8 -4.0 -71.8 -14.8 -71.7 -17.2 C-71.6 -19.6 -28.8 -8.0 -28.6 -12.8 C-28.4 -17.6 -71.5 -44.4 -70.5 -45.9 C-69.5 -47.3 -26.9 -20.7 -22.5 -21.5 C-18.1 -22.3 -45.5 -50.2 -44.2 -50.7 C-42.8 -51.2 -18.3 -19.0 -14.5 -24.6 C-10.7 -30.2 -23.5 -83.6 -21.4 -84.4 C-19.3 -85.2 -7.0 -28.7 -1.9 -29.3 C3.3 -29.9 7.7 -88.2 9.4 -88.0 C11.1 -87.8 1.7 -27.9 8.4 -28.0 C15.1 -28.2 47.9 -89.8 49.6 -88.8 C51.3 -87.9 15.2 -26.7 18.4 -22.2 C21.7 -17.6 67.3 -62.2 69.0 -61.4 C70.7 -60.6 27.7 -23.2 28.7 -17.3 C29.8 -11.4 74.3 -28.0 75.2 -26.0 C76.1 -24.0 34.5 -9.1 34.0 -5.1 C33.5 -1.1 72.5 -3.7 72.1 -2.0Z";
const SPLASH_B =
  "M52.3 1.4 C52.0 3.7 22.8 0.5 27.1 6.9 C31.4 13.4 79.0 38.0 78.1 39.9 C77.3 41.8 28.1 16.9 22.0 18.4 C16.0 20.0 43.6 48.1 42.0 49.1 C40.5 50.2 16.6 22.0 12.9 24.6 C9.2 27.1 21.8 64.5 20.0 64.5 C18.2 64.5 6.6 26.6 1.9 24.4 C-2.9 22.3 -6.4 51.2 -8.5 51.5 C-10.6 51.9 -6.4 26.8 -10.7 26.3 C-15.0 25.8 -32.8 50.1 -34.2 48.6 C-35.6 47.1 -16.8 22.1 -19.3 17.3 C-21.8 12.6 -48.0 22.3 -49.1 20.1 C-50.2 18.0 -20.2 7.8 -26.0 4.3 C-31.7 0.9 -83.2 1.2 -83.6 -0.7 C-83.9 -2.5 -33.1 -1.8 -28.0 -6.6 C-22.9 -11.4 -54.4 -27.6 -53.0 -29.4 C-51.5 -31.2 -19.3 -11.6 -19.4 -17.4 C-19.5 -23.2 -54.6 -62.7 -53.6 -64.2 C-52.6 -65.7 -19.8 -28.0 -13.5 -26.4 C-7.1 -24.8 -17.7 -54.7 -15.7 -54.5 C-13.7 -54.2 -6.1 -22.7 -1.4 -24.9 C3.3 -27.0 10.3 -67.3 12.4 -67.3 C14.4 -67.3 7.9 -29.5 10.8 -25.0 C13.7 -20.5 28.3 -41.7 29.8 -40.4 C31.3 -39.0 14.1 -18.4 19.8 -17.1 C25.5 -15.7 62.3 -34.0 63.8 -32.4 C65.3 -30.7 30.6 -12.8 28.7 -7.2 C26.8 -1.6 52.6 -1.0 52.3 1.4Z";

const RAIN = [4, 11, 17, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93];
const DROPS = [12, 26, 41, 57, 68, 83];
const WAVE = "M0 38 C100 6 100 70 200 38 S300 6 400 38 S500 70 600 38 S700 6 800 38 S900 70 1000 38 S1100 6 1200 38 S1300 70 1400 38 S1500 6 1600 38 L1600 80 L0 80 Z";

export function Hero() {
  return (
    <section className="hero" aria-labelledby="hero-note">
      <div className="hero-art" aria-hidden="true">
        <svg viewBox="0 0 800 320" preserveAspectRatio="xMidYMid slice" focusable="false">
          <defs>
            <linearGradient id="nd-sea" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#050b1e" />
              <stop offset="0.55" stopColor="#0a2a66" />
              <stop offset="1" stopColor="#06627a" />
            </linearGradient>
            <linearGradient id="nd-blood" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#ff4d5e" />
              <stop offset="0.5" stopColor="#e11d48" />
              <stop offset="1" stopColor="#7f1030" />
            </linearGradient>
            <linearGradient id="nd-ice" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#a5f3fc" />
              <stop offset="0.5" stopColor="#22d3ee" />
              <stop offset="1" stopColor="#0e7490" />
            </linearGradient>
            <radialGradient id="nd-halo" cx="0.5" cy="0.5" r="0.5">
              <stop offset="0" stopColor="#e11d48" stopOpacity="0.55" />
              <stop offset="1" stopColor="#e11d48" stopOpacity="0" />
            </radialGradient>
          </defs>
          <rect width="800" height="320" fill="url(#nd-sea)" />
          <circle cx="545" cy="200" r="210" fill="url(#nd-halo)" />
          <g transform="translate(545 205) scale(1.2)">
            <g className="splash-a">
            <path d={SPLASH_A} fill="url(#nd-blood)" />
            <path d={SPLASH_A} fill="none" stroke="#ffd1d8" strokeOpacity="0.35" strokeWidth="1.5" />
        <circle cx="-101.9" cy="73.8" r="5.6" fill="#ff8896" />
        <circle cx="13.7" cy="131.0" r="4.3" fill="#ff8896" />
        <circle cx="59.3" cy="-106.2" r="3.7" fill="#ff8896" />
        <circle cx="-27.7" cy="116.7" r="7.3" fill="#ff8896" />
        <circle cx="108.7" cy="33.8" r="7.9" fill="#ff8896" />
        <circle cx="82.3" cy="-73.2" r="7.0" fill="#ff8896" />
        <circle cx="88.3" cy="69.7" r="6.4" fill="#ff8896" />
        <circle cx="11.4" cy="-107.3" r="7.7" fill="#ff8896" />
        <circle cx="-27.6" cy="-109.0" r="5.9" fill="#ff8896" />
            </g>
          </g>
          <g transform="translate(330 262) scale(0.8)" className="splash-b">
            <path d={SPLASH_B} fill="url(#nd-ice)" fillOpacity="0.8" />
        <circle cx="98.0" cy="50.1" r="6.1" fill="#cffafe" />
        <circle cx="-67.3" cy="-80.6" r="6.9" fill="#cffafe" />
        <circle cx="-98.2" cy="-0.8" r="8.0" fill="#cffafe" />
        <circle cx="84.6" cy="-42.9" r="7.9" fill="#cffafe" />
        <circle cx="16.6" cy="-90.5" r="5.5" fill="#cffafe" />
        <circle cx="25.6" cy="82.5" r="4.0" fill="#cffafe" />
        <circle cx="56.1" cy="65.6" r="3.8" fill="#cffafe" />
        <circle cx="-65.9" cy="-36.6" r="4.5" fill="#cffafe" />
        <circle cx="-43.4" cy="61.6" r="5.2" fill="#cffafe" />
          </g>
        </svg>
        {RAIN.map((l, i) => (
          <i key={l} className="rain" style={{ left: l + "%", animationDelay: (i % 7) * -0.9 + "s", animationDuration: 2.6 + (i % 4) * 0.5 + "s" }} />
        ))}
        {DROPS.map((l, i) => (
          <b key={l} className={"drop " + (i % 2 ? "cy" : "cr")} style={{ left: l + "%", animationDelay: i * -1.3 + "s" }} />
        ))}
        {[["wave w1", 14], ["wave w2", 22], ["wave w3", 34]].map(([c, s]) => (
          <svg key={c} className={c} viewBox="0 0 1600 80" preserveAspectRatio="none" focusable="false" style={{ animationDuration: s + "s" }}>
            <path d={WAVE} />
          </svg>
        ))}
      </div>
      <div className="hero-copy">
        <p className="hero-kicker">Flood susceptibility intelligence</p>
        <p className="hero-title" aria-hidden="true">
          NYANDO FLOOD AI <span>V2</span>
        </p>
        <p className="hero-line">Five wards. One registered model. Every figure traced to a stored run.</p>
        <p id="hero-note" className="hero-note">
          A research view of the evaluation, not a live flood monitor. The artwork is illustration only.
        </p>
      </div>
    </section>
  );
}
