#!/usr/bin/env python3
"""Collect CSE public website feeds and linked documents into a separate dataset.

CSE endpoints are undocumented/reverse-engineered and can change. Raw responses
are retained. This script does not bypass authentication or access controls.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, logging, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests

API = "https://www.cse.lk/api/"
CDN = "https://cdn.cse.lk/"
UA = "StockpulseAcademicResearch/1.0 (polite public-data collection)"
ENDPOINTS = [
    "allSecurityCode", "cntSecurity", "listAllSectors",
    "marketStatus", "marketSummery", "aspiData", "snpData", "allSectors",
    "dailyMarketSummery", "tradeSummary", "todaySharePrice",
    "topGainers", "topLooses", "mostActiveTrades",
    "getNewListingsRelatedNoticesAnnouncements", "getBuyInBoardAnnouncements",
    "approvedAnnouncement", "getFinancialAnnouncement", "circularAnnouncement",
    "directiveAnnouncement", "getNonComplianceAnnouncements",
]
DOC_FEEDS = {
    "approvedAnnouncement": "announcements",
    "getFinancialAnnouncement": "financial_reports",
    "circularAnnouncement": "circulars",
    "directiveAnnouncement": "directives",
    "getNewListingsRelatedNoticesAnnouncements": "listing_notices",
    "getBuyInBoardAnnouncements": "buy_in_board_notices",
    "getNonComplianceAnnouncements": "non_compliance_notices",
}
def now(): return datetime.now(timezone.utc).isoformat()
def safe(s): return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("._") or "item"
def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
def lists(obj):
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list): return v
        for v in obj.values():
            x=lists(v)
            if x is not None: return x
    return []
def flatten_dict(obj, prefix=""):
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items():
            key=f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v,dict): out.update(flatten_dict(v,key))
            elif isinstance(v,list): out[key]=json.dumps(v,ensure_ascii=False)
            else: out[key]=v
    return out
def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        if not keys: f.write(""); return
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()})
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="Dataset/enrichment/cse_live")
    ap.add_argument("--pause",type=float,default=1.0)
    ap.add_argument("--download-documents",action="store_true")
    ap.add_argument("--company-details",action="store_true")
    ap.add_argument("--limit-companies",type=int,default=0)
    args=ap.parse_args()
    out=Path(args.output); raw=out/"raw_json"; out.mkdir(parents=True,exist_ok=True)
    s=requests.Session(); s.headers.update({"User-Agent":UA,"Accept":"application/json, text/plain, */*","Origin":"https://www.cse.lk","Referer":"https://www.cse.lk/"})
    manifest={"source":"https://www.cse.lk/","api_base":API,"started_at_utc":now(),"endpoints":{},"warnings":[
      "CSE API is undocumented and reverse-engineered; endpoint payloads may change.",
      "Current company profile values are snapshots, not historical point-in-time fundamentals.",
      "Report PDFs require extraction, review, and publication-date-aware normalization before ML use."
    ]}
    payloads={}
    for i, ep in enumerate(ENDPOINTS):
        try:
            r=s.post(urljoin(API,ep),timeout=45)
            if r.status_code == 405:
                r=s.get(urljoin(API,ep),timeout=45)
            r.raise_for_status()
            try: data=r.json()
            except ValueError: data={"_non_json_response":r.text[:20000]}
            if isinstance(data, dict) and data.get("apierror", {}).get("status") == "METHOD_NOT_ALLOWED":
                r=s.get(urljoin(API,ep),timeout=45); r.raise_for_status()
                try: data=r.json()
                except ValueError: data={"_non_json_response":r.text[:20000]}
            payloads[ep]=data; dump(raw/(safe(ep)+".json"),data)
            rec_cnt = len(lists(data))
            manifest["endpoints"][ep]={"status":"ok","http_status":r.status_code,"retrieved_at_utc":now(),"record_count":rec_cnt}
            print(f"[{i+1}/{len(ENDPOINTS)}] {ep}: OK ({rec_cnt} records)", flush=True)
        except Exception as e:
            manifest["endpoints"][ep]={"status":"error","error":str(e),"retrieved_at_utc":now()}
            print(f"[{i+1}/{len(ENDPOINTS)}] {ep}: ERROR ({e})", flush=True)
        time.sleep(args.pause)
    # Security master
    securities=lists(payloads.get("allSecurityCode",{}))
    write_csv(out/"securities_master_snapshot.csv",[{**flatten_dict(x),"retrieved_at_utc":now(),"source_url":"https://www.cse.lk/"} for x in securities if isinstance(x,dict)])
    sectors=lists(payloads.get("listAllSectors",{}))
    write_csv(out/"sectors_master_snapshot.csv",[{**flatten_dict(x),"retrieved_at_utc":now()} for x in sectors if isinstance(x,dict)])
    # Disclosure and document manifests; preserve raw fields exactly in JSON as well.
    all_docs=[]
    for ep,kind in DOC_FEEDS.items():
        rows=[]
        for item in lists(payloads.get(ep,{})):
            if not isinstance(item,dict): continue
            row={**flatten_dict(item),"source_endpoint":ep,"document_category":kind,"retrieved_at_utc":now()}
            p=item.get("path") or item.get("filePath") or item.get("documentPath")
            if p:
                p_str = str(p).lstrip("/")
                if not p_str.startswith("cmt/"): p_str = "cmt/" + p_str
                row["document_url"]=urljoin(CDN, p_str)
            rows.append(row); all_docs.append((kind,item,row))
        write_csv(out/f"{kind}_manifest.csv",rows)
    # Per-security current snapshots, only when explicitly requested.
    company_rows=[]; company_errors=[]
    active=[x for x in securities if isinstance(x,dict) and x.get("symbol") and ".N" in str(x.get("symbol")) and str(x.get("active",1)).lower() in ("1","true")]
    if args.limit_companies: active=active[:args.limit_companies]
    if args.company_details:
        print(f"Collecting profiles for {len(active)} active equity symbols...", flush=True)
        for idx, item in enumerate(active):
            symbol=str(item["symbol"])
            try:
                r=s.post(urljoin(API,"companyInfoSummery"),data={"symbol":symbol},timeout=45); r.raise_for_status()
                data=r.json(); dump(raw/"company_profiles"/(safe(symbol)+".json"),data)
                info=data.get("reqSymbolInfo",{}) if isinstance(data,dict) else {}
                beta=data.get("reqSymbolBetaInfo",{}) if isinstance(data,dict) else {}
                company_rows.append({**flatten_dict(item),**flatten_dict(info,"company"),**flatten_dict(beta,"beta"),"retrieved_at_utc":now(),"snapshot_only":True})
            except Exception as e:
                profile_file = raw / "company_profiles" / (safe(symbol) + ".json")
                if profile_file.exists():
                    try:
                        with profile_file.open("r", encoding="utf-8") as pf: data = json.load(pf)
                        info=data.get("reqSymbolInfo",{}) if isinstance(data,dict) else {}
                        beta=data.get("reqSymbolBetaInfo",{}) if isinstance(data,dict) else {}
                        company_rows.append({**flatten_dict(item),**flatten_dict(info,"company"),**flatten_dict(beta,"beta"),"retrieved_at_utc":now(),"snapshot_only":True})
                    except Exception:
                        company_errors.append({"symbol":symbol,"error":str(e),"retrieved_at_utc":now()})
                else:
                    company_errors.append({"symbol":symbol,"error":str(e),"retrieved_at_utc":now()})
            if (idx + 1) % 10 == 0 or (idx + 1) == len(active):
                print(f"[{idx+1}/{len(active)}] Profiles collected ({len(company_rows)} ok, {len(company_errors)} errors)", flush=True)
            time.sleep(args.pause)
    write_csv(out/"company_current_snapshots.csv",company_rows)
    write_csv(out/"company_collection_errors.csv",company_errors)
    # Optional PDF archiving from CSE-provided paths, with host allow-list and SHA256.
    downloaded=[]
    if args.download_documents:
        docs_to_dl = [(k, it, rw) for k, it, rw in all_docs if rw.get("document_url")]
        print(f"Downloading {len(docs_to_dl)} documents with URLs (out of {len(all_docs)} disclosure items)...", flush=True)
        for idx, (kind, item, row) in enumerate(docs_to_dl):
            url=row.get("document_url")
            if not url: continue
            parsed=urlparse(url)
            if parsed.netloc not in ("cdn.cse.lk","www.cse.lk"): continue
            filename=safe(Path(parsed.path).name or "document.pdf")
            docid=safe(item.get("id") or item.get("announcementId") or filename)
            target=out/"documents"/kind/(docid+"_"+filename)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    h=hashlib.sha256(); size=0
                    with s.get(url,stream=True,timeout=60) as resp:
                        resp.raise_for_status()
                        with target.open("wb") as f:
                            for chunk in resp.iter_content(1024*128):
                                if chunk: f.write(chunk); h.update(chunk); size+=len(chunk)
                    row["local_file"]=str(target); row["sha256"]=h.hexdigest(); row["bytes"]=size
                else:
                    row["local_file"]=str(target); row["bytes"]=target.stat().st_size
                downloaded.append(row)
            except Exception as e: row["download_error"]=str(e)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(docs_to_dl):
                print(f"[{idx+1}/{len(docs_to_dl)}] Documents processed ({len(downloaded)} ok, {idx+1-len(downloaded)} errors)", flush=True)
            time.sleep(args.pause)
    write_csv(out/"all_disclosures_document_index.csv",[r for _,_,r in all_docs])
    manifest["finished_at_utc"]=now()
    manifest["security_count"]=len(securities)
    manifest["active_equity_symbols_considered"]=len(active)
    manifest["company_profile_snapshots"]=len(company_rows)
    manifest["document_records"]=len(all_docs)
    manifest["download_documents_enabled"]=args.download_documents
    dump(out/"collection_manifest.json",manifest)
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
