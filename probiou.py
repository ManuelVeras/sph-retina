import torch

def gbb_form(boxes):
    """Transforms bounding box representation into a Generalized B-Box format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing bounding 
                              boxes, where N is the number of boxes and the 
                              columns likely represent (x, y, width, height, angle).

    Returns:
        torch.Tensor: A tensor of the same shape as the input, representing
                      the boxes in the Generalized B-Box format.
    """
    return torch.cat((boxes[:,:2],torch.pow(boxes[:,2:4],2)/12,boxes[:,4:]),1)

def rotated_form(a_, b_, angles):
    """Rotates geometric elements based on provided angles.

    Args:
        a_ (torch.Tensor): Input values representing geometric elements.
        b_ (torch.Tensor): Input values representing geometric elements.
        angles (torch.Tensor): Angles (in radians) for the rotation.

    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): 
            Rotated values for 'a', 'b', and 'c' (the specific meaning 
            of 'c' would need more context).
    """
    a  = a_*torch.pow(torch.cos(angles),2.)+b_*torch.pow(torch.sin(angles),2.)
    b  = a_*torch.pow(torch.sin(angles),2.)+b_*torch.pow(torch.cos(angles),2.)
    c  = a_*torch.cos(angles)*torch.sin(angles)-b_*torch.sin(angles)*torch.cos(angles)
    return a,b,c

def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """Calculates the ProbIoU loss for rotated bounding boxes.

    Args:
        pred (torch.Tensor): Predicted boxes (N, 5), likely in the format
                             (x, y, width, height, angle).
        target (torch.Tensor): Ground truth boxes (N, 5), in the same format 
                               as 'pred'.
        eps (float, optional): Small value for numerical stability. 
                               Defaults to 1e-3.
        mode (str, optional): Loss calculation mode ('l1' or 'l2'). 
                              Defaults to 'l1'.

    Returns:
        torch.Tensor: The calculated ProbIoU loss.
    """

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,0], gbboxes1[:,1], gbboxes1[:,2], gbboxes1[:,3], gbboxes1[:,4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,0], gbboxes2[:,1], gbboxes2[:,2], gbboxes2[:,3], gbboxes2[:,4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = (((a1+a2)*(torch.pow(y1-y2,2)) + (b1+b2)*(torch.pow(x1-x2,2)) )/((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2))+eps))*0.25
    t2 = (((c1+c2)*(x2-x1)*(y1-y2))/((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2))+eps))*0.5
    t3 = torch.log(((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2)))/(4*torch.sqrt((a1*b1-torch.pow(c1,2))*(a2*b2-torch.pow(c2,2)))+eps)+eps)*0.5

    B_d = t1 + t2 + t3

    B_d = torch.clamp(B_d,eps,100.0)
    l1 = torch.sqrt(1.0-torch.exp(-B_d)+eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i+eps)

    if mode=='l1':
       probiou = l1
    if mode=='l2':
       probiou = l2

    return probiou


def main():

    P   = torch.rand(8,5)
    T   = torch.rand(8,5)
    LOSS        = probiou_loss(P, T)
    REDUCE_LOSS = torch.mean(LOSS)
    print(REDUCE_LOSS.item())

if __name__ == '__main__':
    main()