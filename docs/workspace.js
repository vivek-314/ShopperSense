(function() {
  'use strict';
  const $=id=>document.getElementById(id), engine=window.ShopperPredictor;
  const form=$('session-form'), tabs=[...document.querySelectorAll('[role=tab]')];
  let artifact=null, singleResult=null, imported=null, bulkRows=[], page=0, generation=0, historicalSingle=false;
  const PAGE_SIZE=50;
  const fieldIds={product_pages:'product-pages',product_seconds:'product-seconds',information_pages:'information-pages',information_seconds:'information-seconds',account_pages:'account-pages',account_seconds:'account-seconds',visitor_type:'visitor-type',weekend:'weekend'};
  const labels={product_pages:'Product pages viewed',product_seconds:'Time on product pages',information_pages:'Information pages viewed',information_seconds:'Time on information pages',account_pages:'Account / admin pages viewed',account_seconds:'Time on account / admin pages',visitor_type:'Visitor type',weekend:'Weekend visit'};
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function showMessage(id,text) {$(id).textContent=text;$(id).hidden=!text;}
  function switchMode(mode,focus=false) {
    for(const tab of tabs) {
      const active=tab.dataset.mode===mode;
      tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;
      $(tab.getAttribute('aria-controls')).hidden=!active;
      if(active&&focus) tab.focus();
    }
    history.replaceState(null,'',mode==='bulk'?'#csv':location.pathname+location.search);
  }
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>switchMode(tab.dataset.mode));
    tab.addEventListener('keydown',event=>{
      if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key)) {
        event.preventDefault();
        let next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(['ArrowRight','ArrowDown'].includes(event.key)?1:tabs.length-1))%tabs.length;
        switchMode(tabs[next].dataset.mode,true);
      }
    });
  });
  if(location.hash==='#csv') switchMode('bulk');
  // Keep old shared threshold deep links useful after the product homepage replaces the report.
  if(['#threshold','#results','#overview','#preprocessing','#methods','#evaluation','#mistakes','#conclusions','#reproduce','#data'].includes(location.hash)) location.replace('research.html'+location.hash);
  function values() {return Object.fromEntries(new FormData(form));}
  function clearErrors() {
    showMessage('form-error','');
    for(const id of Object.values(fieldIds)) {$(id).removeAttribute('aria-invalid');$(id+'-error').textContent='';}
  }
  function markStale() {
    if(historicalSingle) $('example-label').textContent='Edited historical example · not a real current customer';
    if(singleResult) {
      $('stale-note').hidden=false;$('result-card').classList.add('stale');$('result-status').textContent='INPUTS CHANGED';$('download-single').disabled=true;
    }
  }
  form.addEventListener('input',markStale);form.addEventListener('change',markStale);
  function renderSingle(prediction,name) {
    singleResult={session_id:name||'New session',row_number:1,input:prediction.input,status:'scored',prediction};
    $('empty-result').hidden=true;$('prediction-result').hidden=false;$('stale-note').hidden=true;$('result-card').classList.remove('stale');$('download-single').disabled=false;
    $('result-status').textContent='PREDICTION READY';$('result-session').textContent=name||'New session';
    $('prediction-label').textContent=prediction.predicted_label?'Purchase predicted':'No purchase predicted';
    $('prediction-subtitle').textContent=prediction.predicted_label?'This session falls into the model’s buyer group. Treat it as a review signal, not a confirmed sale.':'This session falls below the model’s purchase cutoff. It can still end in a purchase.';
    const points=prediction.purchase_score*100;
    $('purchase-score').textContent=points.toFixed(1);const unit=document.createElement('small');unit.textContent='/ 100';$('purchase-score').appendChild(unit);
    $('score-fill').style.width=points+'%';$('decision-mark').style.left=(prediction.threshold*100)+'%';
    $('score-track').setAttribute('aria-label',`Purchase signal score ${points.toFixed(1)} out of 100. Decision cutoff ${prediction.threshold*100}.`);
    $('action-title').textContent=prediction.near_boundary?'Borderline: review the context':prediction.predicted_label?'Review this session first':'Give this session a lower review priority';
    $('action-copy').textContent=prediction.near_boundary?'This score is close to the decision cutoff. Small changes in the recorded activity can change the label. Look at your own store’s context before acting.':prediction.predicted_label?'Consider this session earlier when reviewing campaign audiences or consented follow-up. Check your own analytics before spending or contacting anyone.':'Keep this session in your wider audience. This is a prioritisation aid, not a reason to rule out a customer or assume they will never buy.';
    const influences=Object.entries(prediction.contributions).filter(([,value])=>Math.abs(value)>1e-10).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,3);
    const list=$('prediction-signals');list.replaceChildren();
    for(const [field,contribution] of influences) {
      const item=document.createElement('li'),label=document.createElement('span'),impact=document.createElement('b');
      label.textContent=labels[field];
      impact.textContent=artifact.inference.explanation_units==='score points'?`${contribution>0?'+':''}${(100*contribution).toFixed(1)} points`:contribution>0?'Higher signal':'Lower signal';
      if(contribution<0) impact.className='down';item.append(label,impact);list.appendChild(item);
    }
    if(!influences.length) {const item=document.createElement('li');item.textContent='No distinct input influence was recorded for this prediction.';list.appendChild(item);}
    const notes=[...prediction.flags];if(prediction.near_boundary) notes.push('This result is close to the fixed purchase cutoff.');showMessage('prediction-warning',notes.join(' '));
    const test=artifact.test_metrics;
    $('result-caveat').textContent=`On the historical test split, ${test.tp} of ${test.predicted_purchases} flagged sessions actually purchased; ${test.fp} were false alarms. Your store may differ. This score is not a calibrated probability.`;
    $('prediction-time').textContent=`Scored ${new Date().toLocaleString()} · ${artifact.run_id}`;
    $('prediction-result').setAttribute('aria-live','polite');
    $('result-card').scrollIntoView({behavior:'auto',block:'start'});
  }
  function submitSession(event) {
    if(event) event.preventDefault();if(!artifact) return;
    clearErrors();const input=values();
    try {renderSingle(engine.predict(input,artifact),input.session_id.trim());}
    catch(error) {
      showMessage('form-error','Please correct the highlighted fields. No prediction was made for these inputs.');
      for(const item of error.fields||[]) {
        const id=fieldIds[item.field];if(id) {$(id).setAttribute('aria-invalid','true');$(id+'-error').textContent=item.message;}
      }
      const first=error.fields?.[0]&&fieldIds[error.fields[0].field];if(first) $(first).focus();
      if(singleResult) markStale();
    }
  }
  form.addEventListener('submit',submitSession);
  $('clear-session').addEventListener('click',()=>{
    form.reset();clearErrors();singleResult=null;historicalSingle=false;$('example-label').hidden=true;$('empty-result').hidden=false;$('prediction-result').hidden=true;$('stale-note').hidden=true;$('result-card').classList.remove('stale');$('result-status').textContent='WAITING FOR INPUT';$('product-pages').focus();
  });
  function download(name,csv) {
    const url=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $('download-single').addEventListener('click',()=>{if(singleResult&&!$('download-single').disabled) download('shoppersense-session-prediction.csv',engine.exportCSV([singleResult]));});
  async function historicalCSV() {const response=await fetch('assets/example-sessions.csv');if(!response.ok) throw new Error('Historical examples could not be loaded. You can still enter your own session.');return response.text();}
  $('load-example').addEventListener('click',async()=>{
    try {
      const parsed=engine.parseCSV(await historicalCSV()),row=engine.normalize(parsed.records[0].input,artifact).input;
      form.reset();for(const [field,id] of Object.entries(fieldIds)) $(id).value=row[field];$('session-id').value='Historical example';historicalSingle=true;$('example-label').textContent='Historical training example · not a new customer';$('example-label').hidden=false;submitSession();
    } catch(error) {showMessage('form-error',error.message);}
  });
  function resetBulk() {
    generation++;imported=null;bulkRows=[];page=0;$('bulk-results').hidden=true;$('csv-preview').hidden=true;$('score-csv').disabled=false;$('score-csv').hidden=false;$('cancel-csv').hidden=true;$('csv-progress').hidden=true;showMessage('csv-error','');
  }
  function receiveCSV(text,name,historical=false) {
    resetBulk();
    try {
      const parsed=engine.parseCSV(text),invalid=[];
      for(const row of parsed.records) {
        const check=row.parse_error?{valid:false,errors:[{message:row.parse_error}]}:engine.normalize(row.input,artifact);
        if(!check.valid) invalid.push(`Row ${row.row_number}: ${check.errors.map(e=>e.message).join(' ')}`);
      }
      imported=parsed;$('csv-preview').hidden=false;$('csv-filename').textContent=name;
      $('csv-preview-summary').textContent=`${parsed.records.length.toLocaleString()} sessions · ${(parsed.records.length-invalid.length).toLocaleString()} valid · ${invalid.length.toLocaleString()} need fixes`;
      $('csv-column-note').textContent=`All eight input columns recognised.${parsed.ignored_columns.length?' Extra columns ignored: '+parsed.ignored_columns.join(', ')+'.':' No hidden input values will be added.'}`;
      $('csv-example-note').hidden=!historical;
      showMessage('csv-validation',invalid.length?`${invalid.length} row(s) need input fixes. Valid rows can still be scored; invalid rows remain in the result with no score. ${invalid.slice(0,3).join(' ')}${invalid.length>3?' Further errors will be included in the result download.':''}`:'');
    } catch(error) {showMessage('csv-error',error.message);}
  }
  async function receiveFile(file) {
    if(!file||!artifact) return;
    resetBulk();
    if(file.size>5*1024*1024) {showMessage('csv-error','The file is too large. Use a CSV no larger than 5 MB.');return;}
    const token=generation;
    try {const text=await file.text();if(token!==generation) return;receiveCSV(text,file.name);}
    catch {showMessage('csv-error','The file could not be read. Choose a UTF-8 CSV and try again.');}
  }
  $('csv-file').addEventListener('change',event=>receiveFile(event.target.files[0]));
  const drop=$('upload-zone');
  ['dragenter','dragover'].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.add('dragging');}));
  ['dragleave','drop'].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.remove('dragging');}));
  drop.addEventListener('drop',event=>{if(event.dataTransfer.files.length!==1) {showMessage('csv-error','Choose one CSV at a time.');return;}receiveFile(event.dataTransfer.files[0]);});
  $('load-csv-example').addEventListener('click',async()=>{try {receiveCSV(await historicalCSV(),'Five historical training examples',true);} catch(error) {showMessage('csv-error',error.message);}});
  $('score-csv').addEventListener('click',async()=>{
    if(!imported||!artifact) return;
    const current=imported,token=++generation,scored=[];
    $('score-csv').hidden=true;$('cancel-csv').hidden=false;$('csv-progress').hidden=false;$('bulk-results').hidden=true;
    for(let offset=0;offset<current.records.length;offset+=100) {
      if(token!==generation) return;
      scored.push(...engine.scoreRecords({records:current.records.slice(offset,offset+100)},artifact));
      $('csv-progress').textContent=`Scored ${Math.min(offset+100,current.records.length).toLocaleString()} of ${current.records.length.toLocaleString()} session rows…`;
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    if(token!==generation) return;
    bulkRows=scored;page=0;$('result-filter').value='all';$('score-csv').hidden=false;$('cancel-csv').hidden=true;$('csv-progress').textContent='Scoring complete. Every input row is retained in the results.';
    const valid=scored.filter(row=>row.status==='scored'),positive=valid.filter(row=>row.prediction.predicted_label===1);
    $('bulk-scored').textContent=valid.length.toLocaleString();$('bulk-positive').textContent=positive.length.toLocaleString();$('bulk-negative').textContent=(valid.length-positive.length).toLocaleString();$('bulk-invalid').textContent=(scored.length-valid.length).toLocaleString();
    $('bulk-results').hidden=false;renderRows();$('bulk-results').scrollIntoView({behavior:'auto',block:'start'});
  });
  $('cancel-csv').addEventListener('click',()=>{generation++;$('score-csv').hidden=false;$('cancel-csv').hidden=true;$('csv-progress').textContent='Scoring cancelled. No partial list is presented as a complete result.';});
  function renderRows() {
    const filter=$('result-filter').value;
    const filtered=bulkRows.filter(row=>filter==='all'||(filter==='invalid'?row.status==='invalid':row.status==='scored'&&row.prediction.predicted_label===(filter==='positive'?1:0))).sort((a,b)=>a.status==='invalid'&&b.status!=='invalid'?1:b.status==='invalid'&&a.status!=='invalid'?-1:(b.prediction?.purchase_score??-1)-(a.prediction?.purchase_score??-1)||a.row_number-b.row_number);
    const totalPages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE));page=Math.max(0,Math.min(page,totalPages-1));
    const visible=filtered.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE);
    $('bulk-table-body').innerHTML=visible.map(row=>{
      const p=row.prediction,yes=p?.predicted_label===1;
      return `<tr><td>${esc(row.session_id)}<br><small>CSV row ${row.row_number}</small></td><td><span class="prediction-badge ${p?(yes?'':'lower'):'invalid'}">${p?(yes?'Purchase predicted':'No purchase predicted'):'Needs input fixes'}</span></td><td>${p?'<span class="mini-score">'+(p.purchase_score*100).toFixed(1)+'</span>':'—'}</td><td>${p?(p.near_boundary?'Review context':yes?'Review first':'Review later'):'Fix inputs'}</td><td>${esc(row.error||p?.flags.join(' ')||(p?.near_boundary?'Close to the decision cutoff.':'Model prediction; a sale is not confirmed.'))}</td></tr>`;
    }).join('')||'<tr><td colspan="5">No sessions match this filter.</td></tr>';
    $('results-range').textContent=filtered.length?`${page*PAGE_SIZE+1}–${Math.min((page+1)*PAGE_SIZE,filtered.length)} of ${filtered.length.toLocaleString()} sessions`:'0 sessions';
    $('page-label').textContent=`Page ${page+1} of ${totalPages}`;$('previous-page').disabled=page===0;$('next-page').disabled=page>=totalPages-1;
  }
  $('result-filter').addEventListener('change',()=>{page=0;renderRows();});$('previous-page').addEventListener('click',()=>{page--;renderRows();});$('next-page').addEventListener('click',()=>{page++;renderRows();});
  $('download-bulk').addEventListener('click',()=>{if(bulkRows.length) download('shoppersense-session-predictions.csv',engine.exportCSV(bulkRows));});
  async function loadModel() {
    try {
      const response=await fetch('data/predictor.json');if(!response.ok) throw new Error('Model HTTP '+response.status);
      const data=await response.json();
      if(data.schema_version!==1||data.feature_set!=='eight_observable_inputs'||!data.inference||!data.test_metrics||!Array.isArray(data.input_contract)||data.input_contract.length!==7||!Number.isFinite(data.threshold)||data.threshold<0||data.threshold>1) throw new Error('Model format is invalid.');
      artifact=data;
      const t=data.test_metrics;
      $('model-status').textContent='Ready to score your own sessions · Experimental model trained on historical browsing data · No automatic store connection';
      ['predict-button','load-example','load-csv-example'].forEach(id=>$(id).disabled=false);
      $('reliability-found').textContent=`${t.tp} / ${t.purchases}`;$('reliability-flagged').textContent=`${t.tp} / ${t.predicted_purchases}`;
      $('reliability-copy').textContent=`Of ${t.support.toLocaleString()} historical test sessions, the model found ${t.tp} of ${t.purchases} purchases and missed ${t.fn}. It also flagged ${t.fp} sessions that did not purchase. These false alarms make human review essential.`;
      $('model-details').textContent=`${data.model_name}, fitted on ${data.trained_on.toLocaleString()} training sessions with exactly eight inputs. The fixed ${data.threshold.toFixed(2)} score cutoff was selected on validation F1 before this condition’s test evaluation. Its test AP is ${t.ap.toFixed(3)}. It is a separate model from the full-feature research benchmark. Version ${data.run_id}.`;
      $('footer-version').textContent=data.run_id;
    } catch(error) {
      artifact=null;$('model-status').classList.add('error');$('model-status').textContent='The trained model could not be loaded. Reload the page or check your connection. Predictions are disabled; no substitute scores are generated.';console.error('Prediction model unavailable:',error.message);
    }
  }
  loadModel();
})();
