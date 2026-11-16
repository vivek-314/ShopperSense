/* Portable, real inference for fresh inputs. No network, fixture lookup, or synthetic score. */
(function(root) {
  'use strict';
  const aliases = {
    product_pages:['product_pages','ProductRelated'], product_seconds:['product_seconds','ProductRelated_Duration'],
    information_pages:['information_pages','Informational'], information_seconds:['information_seconds','Informational_Duration'],
    account_pages:['account_pages','Administrative'], account_seconds:['account_seconds','Administrative_Duration'],
    visitor_type:['visitor_type','VisitorType'], weekend:['weekend','Weekend'], session_id:['session_id','SessionID','source_row_id']
  };
  function key(value) { return String(value).trim().toLowerCase().replace(/[^a-z0-9]/g,''); }
  function normalize(input, artifact) {
    const result={}, errors=[];
    for (const contract of artifact.input_contract) {
      const name=contract.name, raw=input[name];
      if (raw===undefined || raw===null || String(raw).trim()==='') { errors.push({field:name,message:`${contract.label} is required. Enter 0 only if there was no activity.`}); continue; }
      let numeric;
      if (name==='weekend') {
        const value=String(raw).trim().toLowerCase();
        numeric=['true','yes'].includes(value) ? 1 : ['false','no'].includes(value) ? 0 : /^[01](?:\.0+)?$/.test(value) ? Number(value) : NaN;
      } else {
        const value=String(raw).trim();
        numeric=/^(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?$/i.test(value) ? Number(value) : NaN;
      }
      if (!Number.isFinite(numeric)||numeric<0||(contract.integer&&!Number.isSafeInteger(numeric))||(name==='weekend'&&numeric!==0&&numeric!==1)||numeric>1e9) errors.push({field:name,message:`${contract.label}: use ${name==='weekend'?'Yes or No':contract.integer?'a nonnegative whole number':'nonnegative seconds (a number)'}.`});
      else result[name]=numeric;
    }
    const visitor=String(input.visitor_type??'').trim().toLowerCase();
    const visitors={new:'New_Visitor',new_visitor:'New_Visitor','new visitor':'New_Visitor',returning:'Returning_Visitor',returning_visitor:'Returning_Visitor','returning visitor':'Returning_Visitor',other:'Other'};
    result.visitor_type=visitors[visitor];
    if(!result.visitor_type) errors.push({field:'visitor_type',message:'Choose New visitor, Returning visitor, or Other / unknown.'});
    for(const [pages,seconds] of [['product_pages','product_seconds'],['information_pages','information_seconds'],['account_pages','account_seconds']]) {
      if(result[pages]===0&&result[seconds]>0) errors.push({field:seconds,message:'Time must be 0 when the matching page count is 0.'});
    }
    return {valid:errors.length===0,input:result,errors};
  }
  function predict(input,artifact) {
    const normalized=normalize(input,artifact);
    if(!normalized.valid) { const error=new Error(normalized.errors.map(e=>e.message).join(' ')); error.fields=normalized.errors; throw error; }
    const data=normalized.input, model=artifact.inference;
    const vector=model.numeric_fields.map((field,i)=>model.scale_mean ? (data[field]-model.scale_mean[i])/model.scale_std[i] : data[field]);
    model.categories.forEach(category=>vector.push(data.visitor_type===category?1:0));
    const contributions={};
    let score, base;
    if(model.kind==='logistic') {
      let logit=model.intercept;
      base=1/(1+Math.exp(-model.intercept));
      vector.forEach((value,i)=>{ const delta=value*model.coefficient[i]; logit+=delta; const name=model.transformed_feature_groups[i]; contributions[name]=(contributions[name]||0)+delta; });
      score=logit>=0?1/(1+Math.exp(-logit)):Math.exp(logit)/(1+Math.exp(logit));
    } else {
      const floats=vector.map(Math.fround);
      let total=0, roots=0;
      model.trees.forEach(tree=>{
        let node=0;
        roots+=tree.purchase[0];
        while(tree.left[node]!==-1) {
          const feature=tree.feature[node], next=floats[feature]<=tree.threshold[node]?tree.left[node]:tree.right[node];
          const name=model.transformed_feature_groups[feature];
          contributions[name]=(contributions[name]||0)+(tree.purchase[next]-tree.purchase[node])/model.trees.length;
          node=next;
        }
        total+=tree.purchase[node];
      });
      score=total/model.trees.length;
      base=roots/model.trees.length;
    }
    const flags=artifact.input_contract.filter(c=>c.name!=='weekend'&&data[c.name]>c.training_max).map(c=>`${c.label} exceeds the largest value seen in training (${c.training_max.toLocaleString()}). Treat the result with extra caution.`);
    if(['product_pages','information_pages','account_pages'].every(field=>data[field]===0)) flags.push('No page activity was recorded in these categories. Check that the session export is complete.');
    const near=Math.abs(score-artifact.threshold)<.03;
    return {input:data,purchase_score:score,predicted_label:score>=artifact.threshold?1:0,threshold:artifact.threshold,base_score:base,contributions,flags,near_boundary:near,model_version:artifact.run_id};
  }
  function parseCSV(text) {
    if(typeof text!=='string') throw new Error('Choose a UTF-8 CSV file.');
    if(text.length>5*1024*1024) throw new Error('This browser tool accepts CSV files up to 5 MB.');
    text=text.replace(/^\uFEFF/,'');
    const rows=[]; let row=[], field='', quoted=false, closed=false;
    for(let i=0;i<text.length;i++) {
      const char=text[i];
      if(quoted) {
        if(char==='"'&&text[i+1]==='"') {field+='"';i++;}
        else if(char==='"') {quoted=false;closed=true;}
        else field+=char;
      } else if(char==='"') {
        if(field!==''||closed) throw new Error('Malformed CSV: a quote must start a field.');
        quoted=true;
      } else if(char===','||char==='\n'||char==='\r') {
        row.push(field);field='';closed=false;
        if(char!==',') {if(row.some(value=>value.trim()!=='')) rows.push(row);row=[];if(char==='\r'&&text[i+1]==='\n') i++;}
      } else {
        if(closed&&char.trim()!=='') throw new Error('Malformed CSV: unexpected text after a closing quote.');
        if(!closed) field+=char;
      }
      if(rows.length>10001) throw new Error('Use 10,000 sessions or fewer per file.');
    }
    if(quoted) throw new Error('Malformed CSV: a quoted field was never closed.');
    row.push(field);if(row.some(value=>value.trim()!=='')) rows.push(row);
    if(rows.length<2) throw new Error('The file needs a header and at least one session. The template has headers only; add your session rows.');
    if(rows.length>10001) throw new Error('Use 10,000 sessions or fewer per file.');
    const headers=rows.shift().map(h=>h.trim());
    if(headers.some(h=>!h)||new Set(headers.map(key)).size!==headers.length) throw new Error('Each CSV column needs a unique, nonempty header.');
    const mapping={}, recognized=new Set();
    for(const [field,names] of Object.entries(aliases)) {
      const matches=headers.map((h,i)=>names.some(alias=>key(alias)===key(h))?i:-1).filter(i=>i!==-1);
      if(matches.length>1) throw new Error(`Multiple columns map to ${field}. Keep only one.`);
      if(matches.length===1) {mapping[field]=matches[0];recognized.add(matches[0]);}
    }
    const required=Object.keys(aliases).filter(f=>f!=='session_id');
    const missing=required.filter(f=>mapping[f]===undefined);
    if(missing.length) throw new Error('Missing columns: '+missing.join(', ')+'. Download the template or check the input guide.');
    const records=rows.map((values,i)=>{
      const input={};Object.entries(mapping).forEach(([field,index])=>input[field]=values[index]??'');
      return {row_number:i+2,session_id:String(input.session_id||`Row ${i+2}`),input,parse_error:values.length!==headers.length?`Row has ${values.length} values; expected ${headers.length}.`:null};
    });
    return {records,headers,ignored_columns:headers.filter((_,i)=>!recognized.has(i))};
  }
  function scoreRecords(parsed,artifact) {
    return parsed.records.map(record=>{
      if(record.parse_error) return {...record,status:'invalid',error:record.parse_error};
      try {return {...record,status:'scored',prediction:predict(record.input,artifact)};}
      catch(error) {return {...record,status:'invalid',error:error.message};}
    });
  }
  function csvCell(value) {
    let text=String(value??'');
    if(/^[=+\-@\t\r]/.test(text)) text="'"+text;
    return /[",\n\r]/.test(text)?'"'+text.replace(/"/g,'""')+'"':text;
  }
  function exportCSV(rows) {
    const fields=['session_id','row_number','status','predicted_purchase','purchase_score','review_priority','error','model_version','product_pages','product_seconds','information_pages','information_seconds','account_pages','account_seconds','visitor_type','weekend'];
    const lines=rows.map(row=>{
      const p=row.prediction;
      const values={...row.input,session_id:row.session_id,row_number:row.row_number,status:row.status,predicted_purchase:p?p.predicted_label:'',purchase_score:p?p.purchase_score:'',review_priority:p?(p.predicted_label?'Review first':'Review later'):'Fix inputs',error:row.error||'',model_version:p?p.model_version:''};
      return fields.map(f=>csvCell(values[f])).join(',');
    });
    return fields.join(',')+'\r\n'+lines.join('\r\n')+'\r\n';
  }
  root.ShopperPredictor={predict,normalize,parseCSV,scoreRecords,exportCSV,aliases};
  if(typeof module!=='undefined'&&module.exports) module.exports=root.ShopperPredictor;
})(typeof globalThis!=='undefined'?globalThis:this);
