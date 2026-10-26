import zlib, struct, os, math

def lerp(a,b,t):
    return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))

def render(size):
    S = size
    SS = 4
    N = S*SS
    px = bytearray(N*N*4)
    sky1=(27,23,57); sky2=(96,38,84); sky3=(224,108,90); sky4=(247,178,103)
    skin=(240,194,154); hair=(45,32,27); shirt=(64,190,178); pants=(43,47,66); shoe=(32,27,34)
    def put(x,y,c,a=255):
        i=(y*N+x)*4
        px[i]=c[0]; px[i+1]=c[1]; px[i+2]=c[2]; px[i+3]=a
    def inside_circle(x,y,cx,cy,r):
        return (x-cx)**2+(y-cy)**2 <= r*r
    def inside_rrect(x,y,l,t,r,b,rad):
        if x<l or x>r or y<t or y>b: return False
        cx = min(max(x,l+rad), r-rad)
        cy = min(max(y,t+rad), b-rad)
        return (x-cx)**2+(y-cy)**2 <= rad*rad
    for yy in range(N):
        v=yy/(N-1)
        if v<0.45: col=lerp(sky1,sky2,v/0.45)
        elif v<0.75: col=lerp(sky2,sky3,(v-0.45)/0.30)
        else: col=lerp(sky3,sky4,(v-0.75)/0.25)
        for xx in range(N):
            put(xx,yy,col)
    # sun
    for yy in range(N):
        for xx in range(N):
            if inside_circle(xx,yy,N*0.72,N*0.40,N*0.155):
                put(xx,yy,(255,221,158))
    # distant city
    for i in range(9):
        bx0 = int(N*(0.02+i*0.115)); bw=int(N*0.075); bh=int(N*(0.10+0.05*((i*7)%3))); by=N*0.80
        for yy in range(int(by-bh), int(by)):
            for xx in range(bx0, min(N,bx0+bw)):
                put(xx,yy,(42,27,56))
    # stage
    for yy in range(int(N*0.80), N):
        for xx in range(N):
            t=(yy-N*0.80)/(N*0.20)
            put(xx,yy,lerp((74,51,88),(28,21,37),t))
    for yy in range(int(N*0.79), int(N*0.815)):
        for xx in range(N):
            put(xx,yy,(255,214,133))
    # character
    cx=N*0.5; headr=N*0.135; heady=N*0.37
    # legs
    for yy in range(int(N*0.62), int(N*0.81)):
        for xx in range(int(cx-N*0.11), int(cx-N*0.015)):
            put(xx,yy,pants)
        for xx in range(int(cx+N*0.015), int(cx+N*0.11)):
            put(xx,yy,pants)
    for side in (-1,1):
        l=int(cx+side*N*0.11-(N*0.065 if side>0 else 0)); r=int(cx+side*N*0.11+(N*0.065 if side<0 else 0))
        for yy in range(int(N*0.795), int(N*0.835)):
            for xx in range(l,r):
                put(xx,yy,shoe)
    # body
    for yy in range(int(N*0.47), int(N*0.68)):
        for xx in range(N):
            if inside_rrect(xx,yy,cx-N*0.135,N*0.47,cx+N*0.135,N*0.68,N*0.06):
                put(xx,yy,shirt)
    # arms
    for side in (-1,1):
        ax0 = cx + side*N*0.135; ax1 = cx + side*N*0.21
        l=min(ax0,ax1)-N*0.035; r=max(ax0,ax1)+N*0.035
        for yy in range(int(N*0.49), int(N*0.66)):
            for xx in range(int(l),int(r)):
                if inside_rrect(xx,yy,l,N*0.49,r,N*0.66,N*0.035):
                    put(xx,yy,shirt)
    # head
    for yy in range(N):
        for xx in range(N):
            if inside_circle(xx,yy,cx,heady,headr):
                put(xx,yy,skin)
    # hair
    for yy in range(int(heady-headr), int(heady)):
        for xx in range(int(cx-headr*1.02), int(cx+headr*1.02)):
            if inside_circle(xx,yy,cx,heady,headr*1.04):
                put(xx,yy,hair)
    # fringe
    for yy in range(int(heady-headr*0.55), int(heady-headr*0.1)):
        for xx in range(int(cx-headr), int(cx+headr*0.4)):
            if inside_circle(xx,yy,cx,heady,headr*0.98):
                put(xx,yy,hair)
    # eyes
    for side in (-1,1):
        ex=cx+side*headr*0.38; ey=heady+headr*0.12
        for yy in range(N):
            for xx in range(N):
                if inside_circle(xx,yy,ex,ey,headr*0.11): put(xx,yy,(37,27,34))
    # smile
    for xx in range(int(cx-headr*0.3), int(cx+headr*0.3)):
        t=(xx-(cx-headr*0.3))/(headr*0.6)
        yy=int(heady+headr*0.52+math.sin(t*math.pi)*headr*0.1)
        for d in range(max(1,int(headr*0.05))):
            put(xx,yy+d,(120,60,55))
    # scarf
    for yy in range(int(N*0.465), int(N*0.515)):
        for xx in range(int(cx-N*0.11), int(cx+N*0.11)):
            put(xx,yy,(229,87,91))
    # downsample
    out=bytearray(S*S*4)
    for y in range(S):
        for x in range(S):
            r=g=b=a=0
            for dy in range(SS):
                for dx in range(SS):
                    i=((y*SS+dy)*N+(x*SS+dx))*4
                    r+=px[i]; g+=px[i+1]; b+=px[i+2]; a+=px[i+3]
            n=SS*SS
            i=(y*S+x)*4
            out[i]=r//n; out[i+1]=g//n; out[i+2]=b//n; out[i+3]=a//n
    return out

def write_png(path, size):
    data=render(size)
    raw=b''.join(b'\x00'+bytes(data[y*size*4:(y+1)*size*4]) for y in range(size))
    def chunk(tag,payload):
        c=struct.pack('>I',len(payload))+tag+payload
        return c+struct.pack('>I',zlib.crc32(tag+payload)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',size,size,8,6,0,0,0))
    png+=chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
    open(path,'wb').write(png)
    print(path,size)

base='app/src/main/res'
for folder,size in [('mipmap-mdpi',48),('mipmap-hdpi',72),('mipmap-xhdpi',96),('mipmap-xxhdpi',144),('mipmap-xxxhdpi',192)]:
    write_png(f'{base}/{folder}/ic_launcher.png', size)
os.makedirs('dist',exist_ok=True)
write_png('dist/icon-512.png',512)
