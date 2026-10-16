const $=s=>document.querySelector(s);
const COMP=[['sev','Incident severity',25],['inj','Injuries',20],['back','Unresolved backlog',20],['comp','Complaint pressure',15],['vul','Vulnerability',15],['rec','Recent activity',5],['fc','Forecast (next 90 days)',META.fcw]];
const RC={Good:0,Fair:.33,Poor:.67,Critical:1},PA={Low:0,Medium:.5,High:1};
let allr=[],ranked=[],sel=null,pts=[],TW=1;const W={};

function init(){
  $('#type').innerHTML='<option value="all">All issue types</option>'+TYPES.map(t=>`<option>${t}</option>`).join('');
  $('#zone').innerHTML='<option value="all">All zones</option>'+[...new Set(DATA.map(d=>d.z))].sort().map(z=>`<option>${z}</option>`).join('');
  $('#weights').innerHTML=COMP.map(([k,n])=>`<label>${n}<output id="o_${k}"></output><input type="range" min="0" max="100" id="w_${k}"></label>`).join('');
  const reset=()=>{COMP.forEach(([k,n,w])=>$('#w_'+k).value=w);update(true)};
  COMP.forEach(([k])=>$('#w_'+k).oninput=()=>update(false));
  ['type','zone'].forEach(i=>$('#'+i).onchange=()=>update(true));
  $('#topn').oninput=sim;$('#reset').onclick=reset;$('#export').onclick=exportCSV;
  $('#map').onclick=e=>{const c=$('#map'),b=c.getBoundingClientRect(),x=(e.clientX-b.left)*c.width/b.width,y=(e.clientY-b.top)*c.height/b.height;
    let best=null,bd=144;pts.forEach(p=>{const d=(p.px-x)**2+(p.py-y)**2;if(d<bd){bd=d;best=p.r}});if(best){sel=best;render()}};
  reset();
}
function compute(){
  const type=$('#type').value,zone=$('#zone').value;
  COMP.forEach(([k])=>{W[k]=+$('#w_'+k).value;$('#o_'+k).textContent=W[k]});
  const rows=DATA.map(d=>{const a=[0,0,0,0,0,0];TYPES.forEach((t,i)=>{if(type==='all'||type===t)d.t[i].forEach((v,k)=>a[k]+=v)});
    return{d,n:a[0],hc:a[1],un:a[2],inj:a[3],r:a[5],raw:{sev:a[4],inj:a[3],back:a[2]+d.oc,comp:d.cw,rec:a[5],fc:d.fc,
      vul:d.sc*.3+d.ho*.2+PA[d.pa]*.25+Math.min(d.pd/25000,1)*.15+RC[d.rc]*.1}}});
  const mx={};COMP.forEach(([k])=>mx[k]=Math.max(...rows.map(r=>r.raw[k]))||1);
  TW=COMP.reduce((s,[k])=>s+W[k],0)||1;
  rows.forEach(r=>{r.c={};r.s=0;COMP.forEach(([k])=>{r.c[k]=r.raw[k]/mx[k];r.s+=W[k]*r.c[k]});r.s=100*r.s/TW});
  rows.sort((a,b)=>b.s-a.s);allr=rows;ranked=rows.filter(r=>zone==='all'||r.d.z===zone);
}
const conf=d=>{const e=d.ni+d.cn;return e>=22?'High':e>=15?'Medium':'Low'};
function reasons(r){const d=r.d,t={sev:`${r.hc} High/Critical incident${r.hc==1?'':'s'}`,inj:`${r.inj} injur${r.inj==1?'y':'ies'}`,
  back:`${r.un} unresolved incidents, ${d.oc} open complaints`,comp:`${d.cn} complaints (${d.hp} high priority)`,rec:`${r.r} incidents in last 90 days`,fc:`${d.fc.toFixed(1)} High/Critical expected next 90 days`,
  vul:[d.sc&&'near a school',d.ho&&'near a hospital',d.pa==='High'&&'high pedestrian activity',(d.rc==='Critical'||d.rc==='Poor')&&`road condition ${d.rc}`].filter(Boolean).join(', ')||'average surroundings'};
  return COMP.filter(c=>c[0]!=='fc').map(([k])=>[k,W[k]*r.c[k]]).sort((a,b)=>b[1]-a[1]).slice(0,3).filter(x=>x[1]>0).map(x=>t[x[0]])}
function gaps(r){const g=[],d=r.d;if(d.ni+d.cn<15)g.push('Few records, so this ranking is less certain');if(!r.r)g.push('No incident in last 90 days: fixed, or unreported?');
  if(d.cn>=10&&d.ni<8)g.push('Many complaints but few logged incidents');return g.length?g:['No major gaps flagged']}
const bar=(l,v,m,t)=>`<div class="b"><span>${l}</span><div class="bar"><i style="width:${Math.min(100,v/m*100)}%"></i></div><b>${t??v}</b></div>`;
function render(){
  $('#count').textContent=`${ranked.length} locations · showing top 100`;
  $('#queue tbody').innerHTML=ranked.slice(0,100).map((r,i)=>`<tr data-i="${i}" class="${r===sel?'sel':''}"><td>${i+1}</td><td><b>${r.d.id}</b></td><td>${r.d.z}</td>
    <td><div class="bar"><i style="width:${r.s}%"></i></div>${r.s.toFixed(0)}</td><td><span class="pill ${conf(r.d)}">${conf(r.d)}</span></td><td>${reasons(r).join(' · ')}</td></tr>`).join('');
  document.querySelectorAll('#queue tbody tr').forEach(tr=>tr.onclick=()=>{sel=ranked[+tr.dataset.i];render()});
  detail();drawMap();sim();trend();fair();banner();
}
function detail(){const r=sel;if(!r){$('#detail').innerHTML='<h2>Location</h2><p>No location in this filter.</p>';return}const d=r.d,mx=Math.max(...d.t.map(x=>x[0]),1);
  $('#detail').innerHTML=`<h2>${d.id}<small>${d.z} · ${d.rt} · road ${d.rc}</small></h2><div class="big">${r.s.toFixed(0)}<span>/100 priority</span></div>
  <span class="pill ${conf(d)}">${conf(d)} confidence</span><p class="note">Forecast: <b>${d.fc.toFixed(1)}</b> High/Critical expected in the next 90 days (likely range ${d.fl} to ${d.fh}).</p><h3>Why it ranks here</h3><ul>${reasons(r).map(x=>`<li>${x}</li>`).join('')}</ul>
  <h3>Score breakdown (points)</h3>${COMP.map(([k,n])=>{const p=100*W[k]*r.c[k]/TW;return bar(n,p,40,p.toFixed(1))}).join('')}
  <h3>Incidents by type</h3>${TYPES.map((t,i)=>bar(t,d.t[i][0],mx)).join('')}<h3>Information gaps</h3><ul>${gaps(r).map(x=>`<li>${x}</li>`).join('')}</ul>`}
function drawMap(){const c=$('#map'),x=c.getContext('2d'),p=20;x.clearRect(0,0,c.width,c.height);
  const la=DATA.map(d=>d.la),lo=DATA.map(d=>d.lo),a0=Math.min(...la),a1=Math.max(...la),o0=Math.min(...lo),o1=Math.max(...lo);
  const X=o=>p+(o-o0)/(o1-o0)*(c.width-2*p),Y=a=>c.height-p-(a-a0)/(a1-a0)*(c.height-2*p),ids=new Set(ranked.map(r=>r.d.id)),mx=ranked[0]?ranked[0].s||1:1;pts=[];
  x.fillStyle='#d5dde0';DATA.forEach(d=>{if(!ids.has(d.id)){x.beginPath();x.arc(X(d.lo),Y(d.la),2,0,7);x.fill()}});
  ranked.slice().reverse().forEach(r=>{const t=r.s/mx,px=X(r.d.lo),py=Y(r.d.la);pts.push({r,px,py});x.globalAlpha=.85;x.fillStyle=`hsl(${(1-t)*190},85%,50%)`;x.beginPath();x.arc(px,py,2+t*4,0,7);x.fill()});
  x.globalAlpha=1;if(sel){x.strokeStyle='#000';x.lineWidth=2.5;x.beginPath();x.arc(X(sel.d.lo),Y(sel.d.la),10,0,7);x.stroke()}}
function sim(){const n=+$('#topn').value;$('#nv').textContent=n;const top=ranked.slice(0,n);
  const rows=TYPES.map((t,i)=>{const s=k=>top.reduce((a,r)=>a+r.d.t[i][k],0),tot=DATA.reduce((a,d)=>a+d.t[i][1],0);return{t,hc:s(1),un:s(2),inj:s(3),p:100*s(1)/tot}}).sort((a,b)=>b.hc-a.hc);
  $('#sim').innerHTML=`<table><thead><tr><th>If we fix all issues of this type</th><th>High/Critical incidents addressed</th><th>Share of city total</th><th>Injuries linked</th><th>Unresolved cleared</th></tr></thead><tbody>${
   rows.map(r=>`<tr><td><b>${r.t}</b></td><td>${r.hc}</td><td><div class="bar"><i style="width:${r.p*4}%"></i></div>${r.p.toFixed(1)}%</td><td>${r.inj}</td><td>${r.un}</td></tr>`).join('')}</tbody></table>
   <p class="note">Best single focus for these ${n} locations: <b>${rows[0]?.t}</b>. Based on historical counts, not a forecast.</p>`}
function update(resel){const id=sel&&sel.d.id;compute();sel=(!resel&&ranked.find(r=>r.d.id===id))||ranked[0]||null;render()}

function banner(){$('#banner').innerHTML='<b>Model check:</b> '+META.msg}
function trend(){const c=$('#trend'),x=c.getContext('2d'),z=$('#zone').value,s=ZM[z==='all'?'ALL':z],W=c.width,H=c.height,p=34;x.clearRect(0,0,W,H);
  const mu=s.reduce((a,b)=>a+b,0)/12,sd=Math.sqrt(s.reduce((a,b)=>a+(b-mu)**2,0)/12),m=Math.max(...s)*1.15,X=i=>p+i*(W-2*p)/11,Y=v=>H-p-v/m*(H-2*p);
  x.strokeStyle='#d5dde0';x.lineWidth=1;for(let k=0;k<=4;k++){const y=H-p-k*(H-2*p)/4;x.beginPath();x.moveTo(p,y);x.lineTo(W-p,y);x.stroke()}
  x.strokeStyle='#000';x.lineWidth=2.5;x.beginPath();s.forEach((v,i)=>i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v)));x.stroke();x.font='12px sans-serif';
  s.forEach((v,i)=>{x.fillStyle=v>mu+2*sd?'#d00':'#000';x.beginPath();x.arc(X(i),Y(v),v>mu+2*sd?6:3.5,0,7);x.fill();x.fillStyle='#000';x.fillText('JFMAMJJASOND'[i]+' '+v,X(i)-12,H-10)});
  const d=s.slice(6).reduce((a,b)=>a+b,0)-s.slice(0,6).reduce((a,b)=>a+b,0);
  $('#trendnote').textContent=`${z==='all'?'City':z}: ${s.reduce((a,b)=>a+b,0)} incidents in 2025 · second half vs first half: ${d>=0?'+':''}${d}`}
function fair(){const Z={},P=DATA.reduce((a,d)=>a+d.pd,0),top=allr.slice(0,100);
  DATA.forEach(d=>{const o=Z[d.z]??={n:0,p:0,t:0};o.n++;o.p+=d.pd});top.forEach(r=>Z[r.d.z].t++);
  const rows=Object.entries(Z).map(([z,o])=>({z,n:o.n,t:o.t,st:o.t,sp:100*o.p/P,ratio:(o.t)/(100*o.p/P)})).sort((a,b)=>b.ratio-a.ratio);
  $('#fair').innerHTML='<thead><tr><th>Zone</th><th>Locations</th><th>In top 100</th><th>Share of population density</th><th>Top-100 share ÷ density share</th><th>Flag</th></tr></thead><tbody>'+
   rows.map(r=>{const f=r.ratio>1.5?'over-represented':r.ratio<.5?'under-represented':'';return `<tr class="${r.ratio>1.5?'over':r.ratio<.5?'under':''}"><td>${r.z}</td><td>${r.n}</td><td>${r.t}</td><td>${r.sp.toFixed(1)}%</td><td>${r.ratio.toFixed(2)}</td><td>${f}</td></tr>`}).join('')+'</tbody>'}
function exportCSV(){const q=v=>'"'+String(v).replace(/"/g,'""')+'"',rows=[['rank','location_id','zone','priority','confidence','reasons','information_gaps','forecast_hc_90d']];
  ranked.slice(0,+$('#topn').value).forEach((r,i)=>rows.push([i+1,r.d.id,r.d.z,r.s.toFixed(1),conf(r.d),reasons(r).join('; '),gaps(r).join('; '),r.d.fc.toFixed(2)]));
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(q).join(',')).join('\n')],{type:'text/csv'}));a.download='nextfix_review_list.csv';a.click()}
init();
