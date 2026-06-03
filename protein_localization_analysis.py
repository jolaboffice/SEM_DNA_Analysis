"""SEM-vs-fluorescence PROTEIN localization 3D figures.

Two metrics per modality, both computed internally (no hardcoded values):
  - footprint        = FWHM of the averaged Gaussian PSF peak (resolution).
  - localization
    precision sigma_loc = Gaussian-fit center standard error (covariance-based Cramer-Rao bound;
                          Thompson 2002 Biophys J / Mortensen 2010 Nat Methods), using a
                          measured background-noise model. Depends on both width and SNR.

SEM (fixed)       = Alu EM images; proteins = ML-detected segments. The continuous DNA-backbone
                    signal is subtracted from each cross-section so the peak reflects the protein
                    alone (comparable to the protein-only fluorescence red channel).
Fluorescence      = red-channel spots; only well-isolated spots are used (overlapping spots would
                    bias the footprint). Images of different pixel size are pooled on a common nm grid.

Outputs (300 dpi):
  protein_footprint_3d_{ds}.png    (footprint peaks, same +-320 nm scale)
  protein_precision_zoom_{ds}.png  (zoom +-150 nm; black circle radius = sigma_loc)
"""
import numpy as np, openpyxl
from PIL import Image
from scipy.ndimage import (map_coordinates, gaussian_filter, gaussian_filter1d,
                           shift as ndshift, zoom as ndzoom)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from dna_backbone_trace import compute_line_strength, auto_select_trace
from dna_edge_width import extract_perpendicular_profiles

# ---------- style ----------
BANDS=[0,0.1,0.28,0.46,0.64,0.82,1.0001]
DCMAP=ListedColormap(['#dcdcdc','#264653','#2a9d8f','#8ab17d','#e9c46a','#e76f51'])
DNORM=BoundaryNorm(BANDS,DCMAP.N)

# ---------- parameters ----------
PXs=1000/288.18          # SEM pixel size (nm/px); 288.18 px/um
WIN=320.0                # display half-window (nm)
HW=int(WIN/PXs); HWc=24  # protein patch half-window (transverse/along, px)
HWd=40                   # perpendicular sampling half-width for the trace (px)
ISO_NM=800.0             # fluorescence spot isolation distance (~2x FWHM)
FITHALF=70.0             # SEM along-DNA fit half-window (nm)
ALU=[1,2,3,4,5,6,8,9,10,11]   # SEM Alu images used (Alu7 excluded as outlier) -> 32 segments

gd=lambda x,A,m,s,o:o-A*np.exp(-(x-m)**2/(2*s**2))      # dark feature (SEM)
gb=lambda x,A,m,s,o:o+A*np.exp(-(x-m)**2/(2*s**2))      # bright feature (fluorescence)
g2=lambda xy,A,x0,y0,s,o:(o+A*np.exp(-((xy[0]-x0)**2+(xy[1]-y0)**2)/(2*s**2))).ravel()  # 2D
mad=lambda v:1.4826*np.median(np.abs(v-np.median(v)))   # robust SD

def fit1d_se(x,p,pol,noise,Wfit):
    """Gaussian-fit center standard error (CRLB) for one 1-D profile; x in nm -> returns SE in nm."""
    g=gd if pol=="dark" else gb; ps=gaussian_filter1d(p,1.2)
    ci=int(np.argmin(ps) if pol=="dark" else np.argmax(ps)); mu0=x[ci]
    m=np.abs(x-mu0)<=Wfit; xx,yy=x[m],p[m]
    if len(xx)<7: return None
    base=np.median(yy); amp0=abs((yy.min() if pol=="dark" else yy.max())-base)+1e-6
    try:
        po,cov=curve_fit(g,xx,yy,p0=[amp0,mu0,Wfit/4,base],
            bounds=([0,xx.min(),1,0],[5*amp0,xx.max(),Wfit,yy.max()+1]),
            sigma=np.full(len(xx),noise),absolute_sigma=True,maxfev=6000)
        if np.isfinite(cov[1,1]) and xx.min()<=po[1]<=xx.max(): return np.sqrt(cov[1,1])
    except Exception: pass
    return None

def fwhm2d(Z,axn):
    """FWHM (nm) of a 2-D peak, from its central row (peak normalized to 1)."""
    c=Z[Z.shape[0]//2]; c=c/c.max(); idx=np.where(c>=0.5)[0]
    return (axn[idx[-1]]-axn[idx[0]]) if len(idx)>1 else np.nan

def surf(ax,axn,Z,title,sub,sloc=None,vlim=WIN,box=None):
    """3-D banded surface (contour bands + black wireframe); optional sigma_loc circle above the peak."""
    m=np.abs(axn)<=vlim*1.001; axn=axn[m]; Z=Z[np.ix_(m,m)]
    XX,YY=np.meshgrid(axn,axn)
    ax.plot_surface(XX,YY,Z,cmap=DCMAP,norm=DNORM,shade=False,linewidth=0,antialiased=False,rcount=220,ccount=220)
    ax.plot_wireframe(XX,YY,Z,rcount=26,ccount=26,color='k',linewidth=0.3)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('white'); a.pane.set_edgecolor('0.6'); a.pane.set_alpha(1.0)
    zmax=1.05
    if sloc is not None:                                # point feature: precision = circle of radius sigma_loc
        zc=1.46; zmax=1.66; th=np.linspace(0,2*np.pi,90)
        ax.plot(sloc*np.cos(th),sloc*np.sin(th),zc,color='k',lw=2.4,zorder=12)
        ax.plot([0,0],[0,0],[1.0,zc],color='0.5',lw=0.8,ls=':',zorder=10)
    if box is not None:                                 # top-right info box: sigma_loc + footprint FWHM
        bs,bf=box
        ax.text2D(0.98,0.97,f"$\\sigma_{{loc}}$ = {bs:.1f} nm\nFWHM = {bf:.0f} nm",
                  transform=ax.transAxes,ha='right',va='top',fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.4',fc='white',ec='0.5',lw=0.8))
    ax.set_xlabel("Along DNA contour (nm)",labelpad=14,fontsize=9); ax.set_ylabel("Perpendicular to DNA contour (nm)",labelpad=14,fontsize=9)
    ax.set_zlabel("Normalized signal intensity",labelpad=6,fontsize=9)
    ax.tick_params(axis='both',pad=2)
    ax.set_xlim(-vlim,vlim); ax.set_ylim(-vlim,vlim); ax.set_zlim(0,zmax)
    ax.set_box_aspect((1,1,0.62))
    ax.set_title(f"{title}\n{sub}",pad=2); ax.view_init(elev=20,azim=-60)

# ---------- SEM protein peak + sigma_loc (Alu EM images) ----------
def sem_protein():
    wb=openpyxl.load_workbook('ML_study/predict/predictions.xlsx',read_only=True)
    pacc=[]; pse=[]
    for nm in ['Alu%d'%i for i in ALU]:
        if nm not in wb.sheetnames: continue
        d=list(wb[nm].iter_rows(min_row=2,values_only=True))
        cx=np.array([r[1] for r in d],float); cy=np.array([r[2] for r in d],float); pred=np.array([r[5] for r in d])
        g=np.array(Image.open(f'ML_study/tif/{nm}.tif').convert('L')).astype(float); h,w=g.shape
        ls=compute_line_strength(g); ty,_=auto_select_trace(g,ls,0,h-1,max_step=2,template_sigma=1.5,continuity_weight=1.0,x_avg=1)
        prof,_=extract_perpendicular_profiles(g,np.arange(w).astype(float),ty.astype(float),half_width=HWd)
        inten=map_coordinates(g,[cy,cx],order=1)                 # intensity along the backbone
        s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(cx),np.diff(cy)))])*PXs   # arc length (nm)
        nnoise=mad(inten[pred==0])                               # background noise = non-protein backbone scatter
        runs=[];i=0                                              # contiguous protein segments (ML pred==1)
        while i<len(pred):
            if pred[i]==1:
                j=i
                while j+1<len(pred) and pred[j+1]==1:j+=1
                runs.append((i,j));i=j+1
            else:i+=1
        for a,b in runs:
            c=(a+b)//2
            # localization precision: fit the along-DNA intensity dip
            lo,hi=max(0,c-40),min(len(s),c+41)
            se=fit1d_se(s[lo:hi]-s[c],inten[lo:hi],"dark",nnoise,FITHALF)
            if se: pse.append(se)
            # footprint peak patch: subtract the bare-DNA cross-section -> protein excess only
            tlo,thi=HWd-HWc,HWd+HWc+1
            lo2,hi2=c-HWc,c+HWc+1; patch=np.full((2*HWc+1,2*HWc+1),np.nan)
            s0,s1=max(0,lo2),min(len(prof),hi2); patch[s0-lo2:s1-lo2]=prof[s0:s1,tlo:thi]
            if np.any(np.isnan(patch[HWc-15:HWc+16])): continue
            valid=~np.isnan(patch).any(axis=1); contrast=np.clip(np.median(patch[valid])-patch,0,None); contrast[np.isnan(contrast)]=0.0
            edge=np.array([r for r in range(2*HWc+1) if valid[r] and abs(r-HWc)>15])
            sub=contrast if len(edge)<4 else np.clip(contrast-contrast[edge].mean(axis=0)[None,:],0,None)
            if sub.max()<=0: continue
            pacc.append(sub/sub.max())
    Zsm=gaussian_filter(np.mean(pacc,axis=0),1.8); Zsm=np.clip(Zsm-0.07,0,None); Zsm=(Zsm/Zsm.max()).T
    fwhm=fwhm2d(Zsm,(np.arange(-HWc,HWc+1))*PXs)
    Zp=np.zeros((2*HW+1,2*HW+1)); o=HW-HWc; Zp[o:o+2*HWc+1,o:o+2*HWc+1]=Zsm   # embed into +-320 nm window
    return dict(Z=Zp, ax=(np.arange(-HW,HW+1))*PXs, n=len(pacc),
                sloc=float(np.median(pse)), fwhm=float(fwhm))

# ---------- Fluorescence protein peak + sigma_loc (red spots) ----------
def fluor_protein(items):
    """items=[(image,ppu),...]. Isolated spots only; images of different px pooled on a common nm grid."""
    TGT=np.linspace(-WIN,WIN,49); GX,GY=np.meshgrid(TGT,TGT,indexing='ij'); pts=np.stack([GX.ravel(),GY.ravel()],-1)
    accf=[];ses=[];sig=[];ntot=0
    for nm,ppu in items:
        PXf=1000/ppu; R=int(WIN/PXf)+1; ISO=ISO_NM/PXf
        a=np.array(Image.open(f'wobble/{nm}.tif')).astype(float); h,w=a.shape
        sm=gaussian_filter(a,1.0); noise=mad(a[a<np.percentile(a,40)]); dmax=a.max()
        pk=find_peaks(sm.max(0),prominence=0.20*(sm.max()-sm.min()),distance=6)[0]; ntot+=len(pk)
        for x0 in pk:
            nb=[abs(x0-p) for p in pk if p!=x0]                 # isolation: drop spots with a neighbor < ISO_NM
            if nb and min(nb)<ISO: continue
            y0=int(np.argmax(sm[:,x0])); x1,x2=x0-6,x0+7; y1,y2=y0-6,y0+7
            if x1<0 or y1<0 or x2>w or y2>h: continue
            ys,xs=np.mgrid[y1:y2,x1:x2]; pat=a[y1:y2,x1:x2]; amp=pat.max()-pat.min()
            try:                                                # 2-D Gaussian fit -> center SE (sigma_loc) and width
                po,cov=curve_fit(g2,(xs,ys),pat.ravel(),p0=[amp,x0,y0,2,pat.min()],
                    bounds=([0,x1,y1,0.5,0],[5*amp+1,x2,y2,8,dmax]),sigma=np.full(pat.size,noise),absolute_sigma=True,maxfev=6000)
            except Exception: continue
            if not np.isfinite(cov[1,1]) or x0-R<0 or y0-R<0 or x0+R+1>w or y0+R+1>h: continue
            ses.append(np.sqrt(cov[1,1])*PXf); sig.append(po[3]*PXf)
            pat=a[y0-R:y0+R+1,x0-R:x0+R+1].astype(float); pat=ndshift(pat,(y0-po[2],x0-po[1]),order=1,mode='nearest')
            pat=np.clip(pat-np.percentile(pat,10),0,None)
            coord=(np.arange(-R,R+1))*PXf                       # resample patch onto the common nm grid
            Tg=RegularGridInterpolator((coord,coord),pat,bounds_error=False,fill_value=0.0)(pts).reshape(len(TGT),len(TGT))
            if Tg.max()>0: accf.append(Tg/Tg.max())
    print(f"    fluor: {ntot} detected -> {len(accf)} isolated ({len(items)} images pooled)")
    Zf=ndzoom(np.mean(accf,axis=0),4,order=1); Zf=np.clip(Zf,0,None); Zf=Zf/Zf.max()
    return dict(Z=Zf, ax=np.linspace(-WIN,WIN,Zf.shape[0]), n=len(accf),
                sloc=float(np.median(ses)), fwhm=2.3548*float(np.median(sig)))

# ---------- figures ----------
def make_protein(ds,items,SEM):
    F=fluor_protein(items)
    fig=plt.figure(figsize=(15,6.6))
    surf(fig.add_subplot(1,2,1,projection='3d'),SEM['ax'],SEM['Z'],f"SEM  (n={SEM['n']} proteins)","",box=(SEM['sloc'],SEM['fwhm']))
    surf(fig.add_subplot(1,2,2,projection='3d'),F['ax'],F['Z'],f"Fluorescence: {ds}  (n={F['n']} isolated)","",box=(F['sloc'],F['fwhm']))
    fig.suptitle(f"Protein PSF peak (overlaid at center, same nm scale) — SEM vs {ds}",y=0.99,fontsize=12)
    fig.subplots_adjust(left=0.02,right=0.98,bottom=0.10,top=0.90,wspace=0.05)
    fig.savefig(f"wobble/protein_footprint_3d_{ds}.png",dpi=300,bbox_inches='tight'); plt.close(fig)
    figp=plt.figure(figsize=(14,6))
    surf(figp.add_subplot(1,2,1,projection='3d'),SEM['ax'],SEM['Z'],"SEM protein peak","",sloc=SEM['sloc'],vlim=150,box=(SEM['sloc'],SEM['fwhm']))
    surf(figp.add_subplot(1,2,2,projection='3d'),F['ax'],F['Z'],f"Fluorescence: {ds}","",sloc=F['sloc'],vlim=150,box=(F['sloc'],F['fwhm']))
    figp.suptitle(f"Protein: SEM vs {ds} (zoom ±150 nm); circle = center localization precision $\\sigma_{{loc}}$",y=0.99,fontsize=11)
    figp.subplots_adjust(left=0.02,right=0.98,bottom=0.10,top=0.90,wspace=0.05)
    figp.savefig(f"wobble/protein_precision_zoom_{ds}.png",dpi=300,bbox_inches='tight'); plt.close(figp)
    print(f"[{ds}] SEM: sloc={SEM['sloc']:.2f}nm footprint={SEM['fwhm']:.0f}nm (n={SEM['n']}) | "
          f"fluor: sloc={F['sloc']:.2f}nm footprint={F['fwhm']:.0f}nm (n={F['n']})")

# fluorescence protein (red): Alu4only at 18.75; new C2 set at 15.5 px/um (C2-66 excluded: anomalously bright)
REDS=[('Alu4only_red',18.75),('C2-20_29_4',15.5),('C2-30_35_1',15.5)]
SEM=sem_protein()
print(f"SEM protein: n={SEM['n']} sloc={SEM['sloc']:.2f}nm footprint={SEM['fwhm']:.0f}nm")
print("-- per-image fluorescence (molecule consistency) --")
for nm,ppu in REDS:
    f=fluor_protein([(nm,ppu)]); print(f"   {nm} ({ppu}px/um): n={f['n']} sloc={f['sloc']:.2f}nm FWHM={f['fwhm']:.0f}nm")
make_protein('fluor',REDS,SEM)            # integrated: all red images pooled (pixel-size corrected)
