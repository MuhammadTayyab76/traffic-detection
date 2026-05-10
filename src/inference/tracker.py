class SimpleTracker:
    def __init__(self):
        self.next_id=1
        self.active_tracks=[]

    def update(self,detections):
        updated_tracks=[]
        for det in detections:
            matched=False
            for track in self.active_tracks:
                if(self._iou(det['bbox'],track['bbox'])>0.3):
                    det['track_id']=track['track_id']
                    updated_tracks.append(det)
                    self.active_tracks.remove(track)
                    matched=True
                    break
            if(not matched):
                det['track_id']=self.next_id
                self.next_id+=1
                updated_tracks.append(det)
        self.active_tracks=updated_tracks
        return self.active_tracks

    def _iou(self,boxA,boxB):
        xA=max(boxA[0],boxB[0])
        yA=max(boxA[1],boxB[1])
        xB=min(boxA[2],boxB[2])
        yB=min(boxA[3],boxB[3])
        interArea=max(0,xB-xA)*max(0,yB-yA)
        boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        if(float(boxAArea+boxBArea-interArea)==0):
            return 0
        return interArea/float(boxAArea+boxBArea-interArea)