export default function Hero(){return(
<section className="section min-h-screen flex flex-col justify-center px-6 md:px-16 pt-20 pb-12" style={{background:'linear-gradient(160deg,#060E1A 0%,#0A1628 60%,#0D1E38 100%)'}}>
<div className="flex items-center gap-3 mb-12 fade-up"><span className="w-2 h-2 rounded-full pulse-dot" style={{background:'#C9A84C'}}/><span style={{fontFamily:'IBM Plex Mono',color:'#C9A84C',fontSize:'.7rem',letterSpacing:'.15em',textTransform:'uppercase'}}>Live System - Kisumu, Kenya</span></div>
<h1 className="fade-up delay-1" style={{fontFamily:'Syne,sans-serif',fontSize:'clamp(2rem,8vw,3.5rem)',fontWeight:800,lineHeight:1.1,color:'#F0EAD6',marginBottom:'1.5rem'}}>Protecting<br/><span style={{color:'#C9A84C'}}>161,000 Residents</span><br/>from Flood Risk</h1>
<p className="text-slate-300 text-lg max-w-xl leading-relaxed mb-10 fade-up delay-2">An AI early-warning system for the Nyando Basin combining satellite data, machine learning and real-time predictions to give communities time to act.</p>
<div className="flex flex-wrap gap-4 fade-up delay-3">
<a href="#predict" style={{background:'#C9A84C',color:'#060E1A',fontFamily:'Syne,sans-serif',fontWeight:600,padding:'.75rem 1.75rem',borderRadius:'.25rem',textDecoration:'none'}}>Run a Prediction</a>
<a href="#fund" style={{border:'1px solid rgba(201,168,76,.4)',color:'#C9A84C',fontFamily:'Syne,sans-serif',fontWeight:600,padding:'.75rem 1.75rem',borderRadius:'.25rem',textDecoration:'none'}}>Fund This Project</a>
</div></section>)}