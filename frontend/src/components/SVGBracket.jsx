// frontend/src/components/SVGBracket.jsx
import React, { useMemo } from 'react';

export default function SVGBracket({ bracketData = [], winnerFreq = [], onSelectMatch }) {
  const width = 980;
  const height = 480;

  // Group matches from the bracket data list
  const grouped = useMemo(() => {
    const rounds = { r32: [], r16: [], qf: [], sf: [], final: null };
    
    bracketData.forEach(m => {
      const r = m.round.toLowerCase();
      if (r === 'round of 32') rounds.r32.push(m);
      else if (r === 'round of 16') rounds.r16.push(m);
      else if (r.includes('quarter')) rounds.qf.push(m);
      else if (r.includes('semi')) rounds.sf.push(m);
      else if (r.includes('final')) rounds.final = m;
    });

    return rounds;
  }, [bracketData]);

  // Compute symmetrical positions
  const positions = useMemo(() => {
    const pos = {};
    const midY = height / 2;

    const distY = (count, start, end) => {
      const step = (end - start) / (count - 1 || 1);
      return Array.from({ length: count }, (_, i) => start + i * step);
    };

    const leftR32_Y = distY(8, 20, height - 30);
    const rightR32_Y = distY(8, 20, height - 30);

    // Round of 32
    leftR32_Y.forEach((y, i) => {
      pos[`l-r32-${i}`] = { x: 10, y, match: grouped.r32[i] };
    });
    rightR32_Y.forEach((y, i) => {
      pos[`r-r32-${i}`] = { x: width - 90, y, match: grouped.r32[i + 8] };
    });

    // Round of 16
    for (let i = 0; i < 4; i++) {
      const ly = (pos[`l-r32-${2*i}`].y + pos[`l-r32-${2*i+1}`].y) / 2;
      pos[`l-r16-${i}`] = { x: 120, y: ly, match: grouped.r16[i] };

      const ry = (pos[`r-r32-${2*i}`].y + pos[`r-r32-${2*i+1}`].y) / 2;
      pos[`r-r16-${i}`] = { x: width - 200, y: ry, match: grouped.r16[i + 4] };
    }

    // Quarter Finals
    for (let i = 0; i < 2; i++) {
      const ly = (pos[`l-r16-${2*i}`].y + pos[`l-r16-${2*i+1}`].y) / 2;
      pos[`l-qf-${i}`] = { x: 230, y: ly, match: grouped.qf[i] };

      const ry = (pos[`r-r16-${2*i}`].y + pos[`r-r16-${2*i+1}`].y) / 2;
      pos[`r-qf-${i}`] = { x: width - 310, y: ry, match: grouped.qf[i + 2] };
    }

    // Semi Finals
    const l_sf_y = (pos[`l-qf-0`].y + pos[`l-qf-1`].y) / 2;
    pos[`l-sf-0`] = { x: 340, y: l_sf_y, match: grouped.sf[0] };

    const r_sf_y = (pos[`r-qf-0`].y + pos[`r-qf-1`].y) / 2;
    pos[`r-sf-0`] = { x: width - 420, y: r_sf_y, match: grouped.sf[1] };

    // Final
    pos[`final`] = { x: width / 2 - 40, y: midY + 30, match: grouped.final };

    return pos;
  }, [grouped, width, height]);

  // Generate Bezier connection paths
  const connections = useMemo(() => {
    const paths = [];

    const addCurve = (srcId, dstId, side = 'left') => {
      const src = positions[srcId];
      const dst = positions[dstId];
      if (!src || !dst) return;
      
      const x1 = side === 'left' ? src.x + 80 : src.x;
      const y1 = src.y + 12;
      const x2 = side === 'left' ? dst.x : dst.x + 80;
      const y2 = dst.y + 12;

      const controlX = (x1 + x2) / 2;
      const d = `M ${x1} ${y1} C ${controlX} ${y1}, ${controlX} ${y2}, ${x2} ${y2}`;
      
      const isActive = src.match?.winner && (src.match.winner === dst.match?.team1 || src.match.winner === dst.match?.team2);

      paths.push({ id: `${srcId}-${dstId}`, d, isActive });
    };

    // Connections Left Side
    for (let i = 0; i < 8; i++) addCurve(`l-r32-${i}`, `l-r16-${Math.floor(i/2)}`, 'left');
    for (let i = 0; i < 4; i++) addCurve(`l-r16-${i}`, `l-qf-${Math.floor(i/2)}`, 'left');
    for (let i = 0; i < 2; i++) addCurve(`l-qf-${i}`, `l-sf-0`, 'left');
    addCurve(`l-sf-0`, `final`, 'left');

    // Connections Right Side
    for (let i = 0; i < 8; i++) addCurve(`r-r32-${i}`, `r-r16-${Math.floor(i/2)}`, 'right');
    for (let i = 0; i < 4; i++) addCurve(`r-r16-${i}`, `r-qf-${Math.floor(i/2)}`, 'right');
    for (let i = 0; i < 2; i++) addCurve(`r-qf-${i}`, `r-sf-0`, 'right');
    addCurve(`r-sf-0`, `final`, 'right');

    return paths;
  }, [positions]);

  const renderMatchCard = (id, info) => {
    if (!info || !info.match) return null;
    const m = info.match;
    const isFinished = m.status && m.status !== 'NS';

    return (
      <g 
        key={id} 
        transform={`translate(${info.x}, ${info.y})`}
        className="bracket-node"
        style={{ cursor: 'pointer' }}
        onClick={() => onSelectMatch && onSelectMatch(m.team1, m.team2)}
      >
        <rect
          width="80"
          height="24"
          rx="3"
          fill="rgba(7, 10, 16, 0.95)"
          stroke={isFinished ? 'var(--cyan)' : 'rgba(255,255,255,0.06)'}
          strokeWidth="1"
        />
        {(() => {
          if (isFinished) {
            return (
              <>
                <text x="5" y="10" fill="#fff" fontSize="6px" fontFamily="var(--font-sans)" fontWeight="700">
                  {m.team1_name ? m.team1_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
                <text x="5" y="18" fill="#fff" fontSize="6px" fontFamily="var(--font-sans)" fontWeight="700">
                  {m.team2_name ? m.team2_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
                <text x="75" y="15" fill="var(--cyan)" fontSize="7px" fontFamily="var(--font-display)" fontWeight="800" textAnchor="end">
                  {m.score || 'FT'}
                </text>
              </>
            );
          } else if (m.prediction) {
            const p1_adv = m.prediction.home_win + 0.5 * m.prediction.draw;
            const p2_adv = m.prediction.away_win + 0.5 * m.prediction.draw;
            const isT1Favored = p1_adv >= p2_adv;

            return (
              <>
                <text x="5" y="10" fill={isT1Favored ? '#fff' : 'rgba(255,255,255,0.45)'} fontSize="6px" fontFamily="var(--font-sans)" fontWeight={isT1Favored ? '800' : '600'}>
                  {m.team1_name ? m.team1_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
                <text x="5" y="18" fill={!isT1Favored ? '#fff' : 'rgba(255,255,255,0.45)'} fontSize="6px" fontFamily="var(--font-sans)" fontWeight={!isT1Favored ? '800' : '600'}>
                  {m.team2_name ? m.team2_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
                
                <text x="75" y="10" fill={isT1Favored ? 'var(--cyan)' : 'rgba(0, 242, 254, 0.4)'} fontSize="5.5px" fontFamily="var(--font-display)" fontWeight={isT1Favored ? '800' : '400'} textAnchor="end">
                  {Math.round(p1_adv * 100)}%
                </text>
                <text x="75" y="18" fill={!isT1Favored ? 'var(--cyan)' : 'rgba(0, 242, 254, 0.4)'} fontSize="5.5px" fontFamily="var(--font-display)" fontWeight={!isT1Favored ? '800' : '400'} textAnchor="end">
                  {Math.round(p2_adv * 100)}%
                </text>
              </>
            );
          } else {
            return (
              <>
                <text x="5" y="10" fill="rgba(255,255,255,0.7)" fontSize="6px" fontFamily="var(--font-sans)" fontWeight="700">
                  {m.team1_name ? m.team1_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
                <text x="5" y="18" fill="rgba(255,255,255,0.7)" fontSize="6px" fontFamily="var(--font-sans)" fontWeight="700">
                  {m.team2_name ? m.team2_name.toUpperCase().slice(0, 9) : 'TBD'}
                </text>
              </>
            );
          }
        })()}
      </g>
    );
  };

  const champion = grouped.final?.winner_name;

  // Filter top 10 teams for champion probabilities
  const topWinners = useMemo(() => {
    return winnerFreq.slice(0, 10);
  }, [winnerFreq]);

  // Compute dynamic predictions for Semi-finals and Final based on champion odds
  const predictionsSummary = useMemo(() => {
    if (!grouped.sf || grouped.sf.length < 2 || !grouped.final) {
      return null;
    }
 
    const sf1 = grouped.sf[0];
    const sf2 = grouped.sf[1];
    const fin = grouped.final;
 
    const parseMatchPred = (m) => {
      if (!m || !m.prediction) return null;
      const p = m.prediction;
      const p1_adv = p.home_win + 0.5 * p.draw;
      const p2_adv = p.away_win + 0.5 * p.draw;
      return {
        t1: m.team1_name || 'TBD',
        t2: m.team2_name || 'TBD',
        win1: p.home_win * 100,
        draw: p.draw * 100,
        win2: p.away_win * 100,
        p1_adv: p1_adv * 100,
        p2_adv: p2_adv * 100,
        winner: p1_adv >= p2_adv ? (m.team1_name || 'TBD') : (m.team2_name || 'TBD')
      };
    };
 
    return {
      sf1: parseMatchPred(sf1),
      sf2: parseMatchPred(sf2),
      final: parseMatchPred(fin)
    };
  }, [grouped]);

  return (
    <div className="glass-card">
      <h3 className="card-title">🏆 SIMULATION PLAYGROUND & TOURNAMENT BRACKET</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
        Run simulations to see how neural updates shift the entire World Cup knockout tree and champion win percentages.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.5rem', alignItems: 'start' }}>
        
        {/* Knockout Bracket columns */}
        <div style={{ overflowX: 'auto', background: 'rgba(5,7,12,0.6)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)', padding: '0.5rem' }}>
          <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', minWidth: '650px' }}>
            {/* Symmetrical connected lines */}
            <g>
              {connections.map(c => (
                <path
                  key={c.id}
                  d={c.d}
                  className={`bracket-path ${c.isActive ? 'active' : ''}`}
                />
              ))}
            </g>

            {/* Symmetrical match nodes */}
            <g>
              {Object.entries(positions).map(([id, info]) => renderMatchCard(id, info))}
            </g>

            {/* Champion Display */}
            {champion && (
              <g transform={`translate(${width/2 - 40}, 50)`} className="bracket-node">
                <rect width="80" height="28" rx="4" fill="rgba(245, 166, 35, 0.1)" stroke="var(--gold)" strokeWidth="1.2" />
                <text x="40" y="10" fill="var(--gold)" fontSize="5px" fontFamily="var(--font-display)" fontWeight="800" textAnchor="middle">
                  🏆 CHAMPION
                </text>
                <text x="40" y="21" fill="#fff" fontSize="6.5px" fontFamily="var(--font-sans)" fontWeight="800" textAnchor="middle">
                  {champion.toUpperCase()}
                </text>
              </g>
            )}
          </svg>
        </div>

        {/* Win Probability sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', background: 'rgba(5,7,12,0.6)', border: '1px solid rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px', height: '100%', minHeight: '380px' }}>
          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>🏆 CHAMPION ODDS (TOP 10)</span>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto' }}>
            {topWinners.length > 0 ? (
              topWinners.map((team, idx) => (
                <div key={team.slug} style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem' }}>
                    <span style={{ fontWeight: 700 }}>#{idx + 1} {team.name}</span>
                    <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-display)' }}>{(team.win).toFixed(1)}%</span>
                  </div>
                  <div className="progress-container" style={{ height: '6px' }}>
                    <div 
                      className="progress-bar-fill" 
                      style={{ 
                        width: `${Math.min(100, team.win * 3.5)}%`, 
                        background: 'linear-gradient(90deg, var(--cyan), #0072ff)' 
                      }} 
                    />
                  </div>
                </div>
              ))
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '3rem' }}>
                Awaiting tournament simulation run...
              </div>
            )}
          </div>
        </div>

      </div>

      {/* 🔮 Dynamic Bracket Predictions Summary */}
      {predictionsSummary && (() => {
        const TEAM_FLAGS = {
          'france': '🇫🇷',
          'spain': '🇪🇸',
          'england': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
          'argentina': '🇦🇷',
          'brazil': '🇧🇷',
          'portugal': '🇵🇹',
          'germany': '🇩🇪',
          'netherlands': '🇳🇱',
          'belgium': '🇧🇪',
          'italy': '🇮🇹',
          'colombia': '🇨🇴',
          'uruguay': '🇺🇾',
          'croatia': '🇭🇷',
          'morocco': '🇲🇦',
          'switzerland': '🇨🇭',
          'usa': '🇺🇸',
          'mexico': '🇲🇽',
          'japan': '🇯🇵',
          'canada': '🇨🇦',
          'ecuador': '🇪🇨',
          'senegal': '🇸🇳',
          'australia': '🇦🇺'
        };

        const getFlag = (name) => {
          const slug = name.toLowerCase().replace(/\s+/g, '-').replace('&', 'and');
          return TEAM_FLAGS[slug] || '🏳️';
        };

        return (
          <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(7, 9, 14, 0.4)', border: '1px solid rgba(0, 242, 254, 0.15)', borderRadius: '12px', display: 'flex', flexDirection: 'column', gap: '1rem', boxShadow: 'inset 0 0 15px rgba(0, 242, 254, 0.03)' }}>
            <span style={{ fontSize: '0.85rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, letterSpacing: '1px' }}>
              🔮 STAGE PROJECTIONS & PROBABILITY BREAKDOWN
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
              
              {/* Semi-final 1 */}
              <div style={{
                background: 'rgba(15, 23, 42, 0.45)',
                border: '1px solid rgba(0, 242, 254, 0.08)',
                borderRadius: '10px',
                padding: '1.25rem',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '1rem',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
              }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', letterSpacing: '1px', fontWeight: 600 }}>SEMI-FINAL 1</span>
                    <span style={{ fontSize: '0.65rem', background: 'rgba(0, 242, 254, 0.1)', color: 'var(--cyan)', padding: '2px 8px', borderRadius: '12px', border: '1px solid rgba(0, 242, 254, 0.15)', fontWeight: 600 }}>PRE-MATCH MODEL</span>
                  </div>
                  
                  {/* Team Layout */}
                  {predictionsSummary.sf1 ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '0.5rem 0' }}>
                        <div style={{ flex: 1, textAlign: 'left' }}>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <span>{getFlag(predictionsSummary.sf1.t1)}</span>
                            <span>{predictionsSummary.sf1.t1}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--cyan)' }}>{predictionsSummary.sf1.win1.toFixed(1)}%</strong>
                          </div>
                        </div>
                        <div style={{ padding: '0 0.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--text-secondary)', fontWeight: 800, opacity: 0.4 }}>VS</span>
                          <span style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.4)', marginTop: '0.1rem', whiteSpace: 'nowrap' }}>
                            Draw: {predictionsSummary.sf1.draw.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ flex: 1, textAlign: 'right' }}>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.3rem' }}>
                            <span>{predictionsSummary.sf1.t2}</span>
                            <span>{getFlag(predictionsSummary.sf1.t2)}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--pink)' }}>{predictionsSummary.sf1.win2.toFixed(1)}%</strong>
                          </div>
                        </div>
                      </div>

                      {/* Three-segment Progress Bar */}
                      <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden', margin: '0.75rem 0', display: 'flex' }}>
                        <div style={{ width: `${predictionsSummary.sf1.win1}%`, height: '100%', background: 'var(--cyan)' }} />
                        <div style={{ width: `${predictionsSummary.sf1.draw}%`, height: '100%', background: 'rgba(255,255,255,0.15)' }} />
                        <div style={{ width: `${predictionsSummary.sf1.win2}%`, height: '100%', background: 'var(--pink)' }} />
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '1rem 0', color: 'var(--text-secondary)', fontSize: '0.75rem', textAlign: 'center' }}>
                      Awaiting semi-final 1 setup...
                    </div>
                  )}
                </div>

                {/* Predicted Winner Badge */}
                {predictionsSummary.sf1 && (
                  <div style={{
                    background: 'rgba(0, 242, 254, 0.03)',
                    border: '1px solid rgba(0, 242, 254, 0.15)',
                    borderRadius: '6px',
                    padding: '0.5rem 0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem'
                  }}>
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Predicted Winner</span>
                    <span style={{ color: 'var(--cyan)', fontWeight: 800, fontFamily: 'var(--font-display)' }}>
                      {getFlag(predictionsSummary.sf1.winner)} {predictionsSummary.sf1.winner.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>

              {/* Semi-final 2 */}
              <div style={{
                background: 'rgba(15, 23, 42, 0.45)',
                border: '1px solid rgba(0, 242, 254, 0.08)',
                borderRadius: '10px',
                padding: '1.25rem',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '1rem',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
              }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', letterSpacing: '1px', fontWeight: 600 }}>SEMI-FINAL 2</span>
                    <span style={{ fontSize: '0.65rem', background: 'rgba(0, 242, 254, 0.1)', color: 'var(--cyan)', padding: '2px 8px', borderRadius: '12px', border: '1px solid rgba(0, 242, 254, 0.15)', fontWeight: 600 }}>PRE-MATCH MODEL</span>
                  </div>
                  
                  {/* Team Layout */}
                  {predictionsSummary.sf2 ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '0.5rem 0' }}>
                        <div style={{ flex: 1, textAlign: 'left' }}>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <span>{getFlag(predictionsSummary.sf2.t1)}</span>
                            <span>{predictionsSummary.sf2.t1}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--cyan)' }}>{predictionsSummary.sf2.win1.toFixed(1)}%</strong>
                          </div>
                        </div>
                        <div style={{ padding: '0 0.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--text-secondary)', fontWeight: 800, opacity: 0.4 }}>VS</span>
                          <span style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.4)', marginTop: '0.1rem', whiteSpace: 'nowrap' }}>
                            Draw: {predictionsSummary.sf2.draw.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ flex: 1, textAlign: 'right' }}>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.3rem' }}>
                            <span>{predictionsSummary.sf2.t2}</span>
                            <span>{getFlag(predictionsSummary.sf2.t2)}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--pink)' }}>{predictionsSummary.sf2.win2.toFixed(1)}%</strong>
                          </div>
                        </div>
                      </div>

                      {/* Three-segment Progress Bar */}
                      <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden', margin: '0.75rem 0', display: 'flex' }}>
                        <div style={{ width: `${predictionsSummary.sf2.win1}%`, height: '100%', background: 'var(--cyan)' }} />
                        <div style={{ width: `${predictionsSummary.sf2.draw}%`, height: '100%', background: 'rgba(255,255,255,0.15)' }} />
                        <div style={{ width: `${predictionsSummary.sf2.win2}%`, height: '100%', background: 'var(--pink)' }} />
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '1rem 0', color: 'var(--text-secondary)', fontSize: '0.75rem', textAlign: 'center' }}>
                      Awaiting semi-final 2 setup...
                    </div>
                  )}
                </div>

                {/* Predicted Winner Badge */}
                {predictionsSummary.sf2 && (
                  <div style={{
                    background: 'rgba(0, 242, 254, 0.03)',
                    border: '1px solid rgba(0, 242, 254, 0.15)',
                    borderRadius: '6px',
                    padding: '0.5rem 0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem'
                  }}>
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Predicted Winner</span>
                    <span style={{ color: 'var(--cyan)', fontWeight: 800, fontFamily: 'var(--font-display)' }}>
                      {getFlag(predictionsSummary.sf2.winner)} {predictionsSummary.sf2.winner.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>

              {/* Grand Final */}
              <div style={{
                background: 'rgba(245, 166, 35, 0.02)',
                border: '1px solid rgba(245, 166, 35, 0.25)',
                borderRadius: '10px',
                padding: '1.25rem',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '1rem',
                boxShadow: '0 4px 25px rgba(245, 166, 35, 0.06)'
              }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--gold)', fontFamily: 'var(--font-display)', letterSpacing: '1.2px', fontWeight: 800 }}>🏆 GRAND FINAL</span>
                    <span style={{ fontSize: '0.65rem', background: 'rgba(245, 166, 35, 0.15)', color: 'var(--gold)', padding: '2px 8px', borderRadius: '12px', border: '1px solid rgba(245, 166, 35, 0.3)', fontWeight: 800 }}>PRE-MATCH MODEL</span>
                  </div>
                  
                  {/* Team Layout */}
                  {predictionsSummary.final ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '0.5rem 0' }}>
                        <div style={{ flex: 1, textAlign: 'left' }}>
                          <div style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                            <span>{getFlag(predictionsSummary.final.t1)}</span>
                            <span>{predictionsSummary.final.t1}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--cyan)' }}>{predictionsSummary.final.win1.toFixed(1)}%</strong>
                          </div>
                        </div>
                        <div style={{ padding: '0 0.75rem', fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--gold)', fontWeight: 800, opacity: 0.6 }}>VS</div>
                        <div style={{ flex: 1, textAlign: 'right' }}>
                          <div style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.35rem' }}>
                            <span>{predictionsSummary.final.t2}</span>
                            <span>{getFlag(predictionsSummary.final.t2)}</span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                            Win: <strong style={{ color: 'var(--gold)' }}>{predictionsSummary.final.win2.toFixed(1)}%</strong>
                          </div>
                        </div>
                      </div>

                      {/* Three-segment Progress Bar */}
                      <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden', margin: '0.75rem 0', display: 'flex' }}>
                        <div style={{ width: `${predictionsSummary.final.win1}%`, height: '100%', background: 'var(--cyan)' }} />
                        <div style={{ width: `${predictionsSummary.final.draw}%`, height: '100%', background: 'rgba(255,255,255,0.15)' }} />
                        <div style={{ width: `${predictionsSummary.final.win2}%`, height: '100%', background: 'var(--gold)' }} />
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '1rem 0', color: 'var(--text-secondary)', fontSize: '0.75rem', textAlign: 'center' }}>
                      Awaiting grand final setup...
                    </div>
                  )}
                </div>

                {/* Predicted Champion Badge */}
                {predictionsSummary.final && (
                  <div style={{
                    background: 'rgba(245, 166, 35, 0.06)',
                    border: '1px solid rgba(245, 166, 35, 0.35)',
                    borderRadius: '6px',
                    padding: '0.5rem 0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem'
                  }}>
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 700 }}>Projected Champion</span>
                    <span style={{ color: 'var(--gold)', fontWeight: 800, fontFamily: 'var(--font-display)', textShadow: '0 0 8px rgba(245, 166, 35, 0.4)' }}>
                      {getFlag(predictionsSummary.final.winner)} {predictionsSummary.final.winner.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>

            </div>
          </div>
        );
      })()}
    </div>
  );
}
