def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    """https://github.com/hysts/pytorch_cutout"""
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def hide_and_seek(img):
    """https://github.com/kkanshul/Hide-and-Seek"""
    # get width and height of the image
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_prob = 0.5
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size<>0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                       img[x:x_end,y:y_end,:]=0

    return img



class GridMask1(nn.Module):
    """https://github.com/dvlab-research/GridMask"""
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        #d = self.d
        #self.l = int(d*self.ratio+0.5)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
#        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 

        return x.view(n,c,h,w)
    
class GridMask2(nn.Module):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n,c,h,w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n,c,h,w)
        return y