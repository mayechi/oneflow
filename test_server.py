from collections import deque

import numpy as np
import sys
import torch
import torch.nn.functional as F
import time
import cv2
import lap
import pdb
import json

from .utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms
from static_3d import Static_3d
sys.path.append("..") 
from EKF import Kalman_Filter

class Tracker:
	"""The main tracking file, here is where magic happens."""
	dts = 0.04
	monitor_flag = 2
	if monitor_flag == 0:       
    	# tracking in Driving recorder
		cl_bus = 1
		cl_car = 2
		cl_person = 4#4
		cl_truck = 9
		cl_car_list = [cl_bus, cl_car, cl_truck]
		cl_person_list = [cl_person]
	elif monitor_flag == 1:
        # tracking in monitor
		cl_car = 1
		cl_person = 2
		cl_Biker = 3
		cl_motor = 4
		cl_truck = 5
		cl_car_truck_list = [cl_car, cl_truck]
		cl_car_truck_motor_list = [cl_car, cl_truck, cl_motor]        
		cl_motor_bike_person_list = [cl_motor, cl_Biker, cl_person]
        
		cl_car_list = [cl_car, cl_truck, cl_motor]
		cl_person_list = [cl_person, cl_Biker]   
	elif monitor_flag == 2:
        # tracking in lab   
		cl_person = 2
		cl_car_truck_list = []
		cl_car_truck_motor_list = []        
		cl_motor_bike_person_list = [cl_person]        
		cl_car_list = []
		cl_person_list = [cl_person]
	elif monitor_flag == 3:
        # tracking in lab   
		cl_car = 1
		cl_truck = 5
		cl_car_truck_list = [cl_car, cl_truck]
		cl_car_truck_motor_list = [cl_car, cl_truck]        
		cl_motor_bike_person_list = []        
		cl_car_list = [cl_car, cl_truck]
		cl_person_list = []
	elif monitor_flag == 4:
        # tracking in out lab   
		cl_person = 2
		cl_Biker = 3        
		cl_car_truck_list = []
		cl_car_truck_motor_list = []        
		cl_motor_bike_person_list = [cl_person, cl_Biker]        
		cl_car_list = []
		cl_person_list = [cl_person, cl_Biker]
	def __init__(self, obj_detect, reid_network_person, reid_network_car, tracker_cfg, camera_id, region_file, mapping_array):
		self.obj_detect = obj_detect
		self.reid_network_person = reid_network_person
		self.reid_network_car = reid_network_car
		self.detection_car_thresh = tracker_cfg['detection_car_thresh']
		self.detection_person_thresh = tracker_cfg['detection_person_thresh']
		self.track_new_car_thresh = tracker_cfg['track_new_car_thresh']
		self.track_new_person_thresh = tracker_cfg['track_new_person_thresh']
		self.regression_person_thresh = tracker_cfg['regression_person_thresh']
		self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
		self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
		self.iou_thresh = tracker_cfg['iou_thresh']
		self.public_detections = tracker_cfg['public_detections']
		self.inactive_patience = tracker_cfg['inactive_patience']
		self.do_reid = tracker_cfg['do_reid']
		self.max_features_num = tracker_cfg['max_features_num']
		self.reid_person_sim_threshold = tracker_cfg['reid_person_sim_threshold']
		self.reid_car_sim_threshold = tracker_cfg['reid_car_sim_threshold']
		self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
		self.do_align = tracker_cfg['do_align']
		self.motion_model_cfg = tracker_cfg['motion_model']

		self.warp_mode = eval(tracker_cfg['warp_mode'])
		self.number_of_iterations = tracker_cfg['number_of_iterations']
		self.termination_eps = tracker_cfg['termination_eps']
		self.continuous_frame_threshold = tracker_cfg['continuous_frame_threshold']
        
		self.camera_id = camera_id
#		self.region_car_json = region_file+'region.json'
#		self.region_person_json = region_file+'region.json'
#		with open(self.region_car_json) as f:
#		    region_dict = json.loads(f.read())
#		    self.region_car = np.array(region_dict['region'])
#		with open(self.region_person_json) as f:
#		    region_dict = json.loads(f.read())
#		    self.region_person = np.array(region_dict['region'])

		self.region_json = region_file+'region_acc.json'
		with open(self.region_json) as f:
		    region_dict = json.loads(f.read())
		    self.region = np.array(region_dict['region'])
		self.mapping_array = mapping_array
		self.tracks = []
		self.inactive_tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}
		self.results_edit = {} 
		self.results_2d = {}
		self.results_3d = {}
		self.kal_dict = {}
		f = 24
		cameraMatrix = np.array([[2741.791318135484, 0, 2011.321500928017], 
                                 [0, 2758.260795485328, 1076.472607823354],
                                 [0, 0, 1]])
		rt = np.array([[0.998479, -0.0536106, 0.0128535, 481.141],
                       [-0.00434605, -0.308968, -0.951063, 10920.2],
                       [0.0549584, 0.94956, -0.308731, 4583.91]])  
		xyz3 = [(-211.902100, -8514.875977, -23.216675), (-1223.614258, -2564.342285, -2.237671), (1412.399658, -4786.003418, -19.095947)]        
		self.static_obj = Static_3d(f, cameraMatrix, rt, xyz3)

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.continuous_frame = 0
			t.pos = t.last_pos[-1]
		self.inactive_tracks += tracks

	def add(self, blob, new_det_pos, new_det_scores, new_det_clses, new_det_features):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		num = 0
		for i in range(num_new):
#			if int(new_det_clses[i].item()) == 4:
			if int(new_det_clses[i].item()) in self.cl_person_list:
				if new_det_scores[i] > self.track_new_person_thresh:                
					self.tracks.insert(0, Track(
        				new_det_pos[i].view(1, -1),
        				new_det_scores[i],
                        new_det_clses[i],
        				str(self.track_num+num),#str(self.camera_id)+'-'+str(self.track_num+i),
        				int(blob['idx']),                
        				new_det_features[i].view(1, -1),
        				self.inactive_patience,
        				self.max_features_num,
        				self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                        0
        			))
					num += 1
			else:
				if new_det_scores[i] > self.track_new_car_thresh:                     
					self.tracks.append(Track(
        				new_det_pos[i].view(1, -1),
        				new_det_scores[i],
                        new_det_clses[i],
        				str(self.track_num+num),#str(self.camera_id)+'-'+str(self.track_num+i),
        				int(blob['idx']),                
        				new_det_features[i].view(1, -1),
        				self.inactive_patience,
        				self.max_features_num,
        				self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1,
                        0
        			))
					num += 1
		self.track_num += num
#		print("track_num:", self.track_num)
	def compute_iou(self, rec1, rec2):
#		rec1 = (rec1[1], rec1[0], rec1[1]+rec1[3], rec1[0]+rec1[2])
#		rec2 = (rec2[1], rec2[0], rec2[1]+rec2[3], rec2[0]+rec2[2])
		rec1 = (rec1[1], rec1[0], rec1[3], rec1[2])
		rec2 = (rec2[1], rec2[0], rec2[3], rec2[2])        
		"""
		computing IoU
		:param rec1: (y0, x0, y1, x1), which reflects
		        (top, left, bottom, right)
		:param rec2: (y0, x0, y1, x1)
		:return: scala value of IoU
		"""
		# computing area of each rectangles
		S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
		S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
     
		# computing the sum_area
		sum_area = S_rec1 + S_rec2
     
		# find the each edge of intersect rectangle
		left_line = max(rec1[1], rec2[1])
		right_line = min(rec1[3], rec2[3])
		top_line = max(rec1[0], rec2[0])
		bottom_line = min(rec1[2], rec2[2])
     
		# judge if there is an intersect
		if left_line >= right_line or top_line >= bottom_line:
			return 0
		else:
			intersect = (right_line - left_line) * (bottom_line - top_line)
		return (intersect / (sum_area - intersect)) * 1.0
	def match_find_max(self, ori):
		temp = []
		temp_score = []
		for i in range(len(ori)):
			col_argmax = np.argmax(ori[i])
			col_max = np.max(ori[i])
			row_argmax = np.argmax(ori[:, col_argmax])
			row_max = np.max(ori[:, col_argmax])
			if i == row_argmax and col_max != 0 and row_max != 0:
				temp.append(col_argmax)
				temp_score.append(col_max)
			else:
				temp.append(-1)
				temp_score.append(-1)
		return temp, temp_score
	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		pos = self.get_pos()
		# regress
		boxes, scores = self.obj_detect.predict_boxes(blob['img'], pos)
		pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

		s = []
		for i in range(len(self.tracks) - 1, -1, -1):
			t = self.tracks[i]
			t.score = scores[i]
			if scores[i] <= self.regression_person_thresh:
				self.tracks_to_inactive([t])
			else:
				s.append(scores[i])
				t.pos = pos[i].view(1, -1)

		return torch.Tensor(s[::-1]).cuda()

	def regress_tracks_mm(self, blob, det_boxes, det_scores, det_clses):
		"""Regress the position of the tracks and also checks their scores."""
		score_mat = np.zeros([len(self.tracks), det_boxes.shape[0]])
		s = []
		c = []
#		t8 = time.time()
		for i in range(len(self.tracks)-1, -1, -1):
			t = self.tracks[i]
			if i <= len(self.tracks)/2:
				range_det_boxes = range(det_boxes.shape[0])
			else:
				range_det_boxes = range(det_boxes.shape[0]-1, -1, -1)
			for j in range_det_boxes:
#				t12 = time.time()
				score  = self.compute_iou(t.pos[0], det_boxes[j])     
#				t13 = time.time()
#				print('t13-t12:', t13-t12)
				if score > 0.8:
					score_mat[i, j] = score
					break

#		score_mat_ori = score_mat.copy()

        
        # lap method for match     
		cost, x, y = lap.lapjv(1-score_mat, extend_cost=True, cost_limit=self.iou_thresh)
		disappear_track  = []
		new_track  = []
		result = []
		for i, j in enumerate(x):
			if j == -1:
			  disappear_track.append(i)  
			else:
			  result.append([i, j, 1.0]) 
		for i, j in enumerate(y):
			if j == -1:
			  new_track.append(i)  
#		result = [[i, j, 1.0] for i, j in enumerate(x) if j != -1]
            
        # My method for match
#		result = []
#		while (score_mat == 0).all() == False:
#			temp, temp_score = self.match_find_max(score_mat)
#			for i in range(len(temp)):
#				if temp[i] != -1:
#					result.append([i, temp[i], temp_score[i]])
#					score_mat[i, :] = 0
#					score_mat[:, temp[i]] = 0
		for i in range(len(self.tracks)-1, -1, -1):
			t = self.tracks[i]
			flag_match = 0
			for j in range(len(result)-1, -1, -1):
				if i == result[j][0]:
					flag_match = 1
					if result[j][2] > self.iou_thresh:
						t.score = det_scores[result[j][1]]
						t.cls = det_clses[result[j][1]]
						t.pos = det_boxes[result[j][1]].view(1, -1)
						t.frame_id = int(blob['idx'])
						t.continuous_frame += 1
						s.append(t.score)
						c.append(t.cls)
					else:
						self.tracks_to_inactive([t])
					break
			if flag_match == 0:
				self.tracks_to_inactive([t])
#		t11 = time.time()
#		print('t9-t8', t9-t8)
#		print('t10-t9', t10-t9)
#		print('t11-t10', t11-t10)
		return torch.Tensor(s[::-1]).cuda(), torch.Tensor(c[::-1]).cuda(), disappear_track, new_track

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks], 0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def get_cls(self):
		"""Get the clses of all active tracks."""
		if len(self.tracks) == 1:
			cls = self.tracks[0].cls
		elif len(self.tracks) > 1:
			cls = torch.cat([t.cls for t in self.tracks], 0)
		else:
			cls = torch.zeros(0).cuda()
		return cls

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def reid(self, blob, new_det_pos, new_det_scores, new_det_clses, new_features):
		"""Tries to ReID inactive tracks with provided detections."""
		temp = 0
		for i in range(len(new_det_clses)):
			if int(new_det_clses[i].item()) in self.cl_person_list:
				temp = i+1         
		if self.do_reid:
			new_det_features = new_features
#			if temp == 0:
#				new_det_features = self.reid_network_car.test_rois(
#    				blob['img_ori'], new_det_pos).data   
#			elif temp == len(new_det_pos):
#				new_det_features = self.reid_network_person.test_rois(
#    				blob['img_ori'], new_det_pos).data                 
#			else:
#				new_det_person_features = self.reid_network_person.test_rois(
#    				blob['img_ori'], new_det_pos[0:temp]).data                  
#				new_det_car_features = self.reid_network_car.test_rois(
#    				blob['img_ori'], new_det_pos[temp:]).data
#				new_det_features = torch.cat((new_det_person_features, new_det_car_features), 0)            
			if len(self.inactive_tracks) >= 1:
				# calculate appearance distances
				dist_mat, pos = [], []
				for t in self.inactive_tracks:
					temp = []
					for i in range(len(new_det_features)):
				 		if (t.cls in self.cl_car_list and new_det_clses[i] in self.cl_car_list) or \
                           (t.cls in self.cl_person_list and new_det_clses[i] in self.cl_person_list):
				 			temp.append(t.test_features_cosine(new_det_features[i]).view(1, -1))
				 		else:
				 			temp.append(torch.tensor(0.0).view(1, -1).cuda())                             
					dist_mat.append(torch.cat(temp, dim=1))                             
					pos.append(t.pos)
				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
					pos = torch.cat(pos, 0)
				else:
					dist_mat = dist_mat[0]
					pos = pos[0]

				# calculate IoU distances
				iou = bbox_overlaps(pos, new_det_pos)
				iou_mask = torch.ge(iou, self.reid_iou_threshold)
				iou_neg_mask = ~iou_mask
				# make all impossible assignments to the same add big value
				dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
				dist_mat = dist_mat.cpu().numpy()
#				dist_mat_ori= dist_mat.copy()


                # lap method for match     
				cost, x, y = lap.lapjv(1-dist_mat, extend_cost=True, \
                           cost_limit=min(self.reid_person_sim_threshold, self.reid_car_sim_threshold))
				result = [[i, j, dist_mat[i, j]] for i, j in enumerate(x) if j != -1]                
                
                # My method for match
#				result = []                
#				while (dist_mat == 0).all() == False:
#					temp, temp_score = self.match_find_max(dist_mat)
#					for i in range(len(temp)):
#						if temp[i] != -1:
#							result.append([i, temp[i], temp_score[i]])
#							dist_mat[i, :] = 0
#							dist_mat[:, temp[i]] = 0                
				
				assigned = []
				remove_inactive = []
				for temp in result:
					r, c, s = temp[0], temp[1], temp[2]
					t = self.inactive_tracks[r]
					if t.cls.item() in self.cl_person_list:
						self.reid_sim_threshold = self.reid_person_sim_threshold
					else:
						self.reid_sim_threshold = self.reid_car_sim_threshold
					if s >= self.reid_sim_threshold:
						r, c = temp[0], temp[1]
						t = self.inactive_tracks[r]
#						if int(t.cls.item()) == 4:
						if int(t.cls.item()) in self.cl_person_list:
							self.tracks.insert(0, t)
						else:
							self.tracks.append(t)
						t.count_inactive = 0
						t.pos = new_det_pos[c].view(1, -1)
						t.score = new_det_scores[c]
						t.cls = new_det_clses[c]
						t.frame_id = int(blob['idx'])
						t.reset_last_pos()
						t.add_features(new_det_features[c].view(1, -1))
						assigned.append(c)
						remove_inactive.append(t)                        

				for t in remove_inactive:
					self.inactive_tracks.remove(t)

				keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
				if keep.nelement() > 0:
					new_det_pos = new_det_pos[keep]
					new_det_scores = new_det_scores[keep]
					new_det_clses = new_det_clses[keep]
					new_det_features = new_det_features[keep]
				else:
					new_det_pos = torch.zeros(0).cuda()
					new_det_scores = torch.zeros(0).cuda()
					new_det_clses = torch.zeros(0).cuda()
					new_det_features = torch.zeros(0).cuda()
		
		return new_det_pos, new_det_scores, new_det_clses, new_det_features

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		poses = self.get_pos()
		clses = self.get_cls()
		temp = 0
		for i in range(len(clses)):
#			if int(clses[i].item()) == 4:
			if int(clses[i].item()) in self.cl_person_list:
				temp = i+1
		if temp == 0:
#			t_reid_0 = time.time()
			new_features = self.reid_network_car.test_rois(blob['img_ori'], poses).data
#			t_reid_1 = time.time()     
#			print("t_reid_1-t_reid_0:", t_reid_1-t_reid_0)
		elif temp == len(clses):
			t_reid_0 = time.time()            
			new_features = self.reid_network_person.test_rois(blob['img_ori'], poses).data
			t_reid_1 = time.time()             
			print("t_reid_1-t_reid_0:", t_reid_1-t_reid_0)
		else:
			new_person_features = self.reid_network_person.test_rois(blob['img_ori'], poses[0:temp]).data
			new_car_features = self.reid_network_car.test_rois(blob['img_ori'], poses[temp:]).data
			new_features = torch.cat((new_person_features, new_car_features), 0)
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t, f in zip(self.tracks, new_features):
			t.add_features(f.view(1, -1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
			im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
			im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
			im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
			cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)

			for t in self.tracks:
				t.pos = warp_pos(t.pos, warp_matrix)
				# t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					t.pos = warp_pos(t.pos, warp_matrix)

			if self.motion_model_cfg['enabled']:
				for t in self.tracks:
					for i in range(len(t.last_pos)):
						t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

	def motion_step(self, track):
		"""Updates the given track's position by one step based on track.last_v"""
		if self.motion_model_cfg['center_only']:
			center_new = get_center(track.pos) + track.last_v
			track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
		else:
			track.pos = track.pos + track.last_v

	def motion(self):
		"""Applies a simple linear motion model that considers the last n_steps steps."""
		for t in self.tracks:
			last_pos = list(t.last_pos)

			# avg velocity between each pair of consecutive positions in t.last_pos
			if self.motion_model_cfg['center_only']:
				vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
			else:
				vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

			t.last_v = torch.stack(vs).mean(dim=0)
			self.motion_step(t)

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					self.motion_step(t)
	def mapping_results(self):
	    for i, ob_id in enumerate(self.results):
	        for j, frame_id in enumerate(self.results[ob_id]):
	            if ob_id not in self.results_3d:
	                self.results_3d[ob_id] = {}
	                self.results_2d[ob_id] = {}                    
	            bbox = self.results[ob_id][frame_id]
	            score = float(self.results[ob_id][frame_id][4])
	            cls = float(self.results[ob_id][frame_id][5])
	            c_f = float(self.results[ob_id][frame_id][6])
#	            temp_0 = []                
#	            temp_1 = []
#	            temp_2 = []
	            temp_0 = [self.cl_person]                
	            temp_1 = [self.cl_car, self.cl_Biker, self.cl_motor]
	            temp_2 = [self.cl_truck]
	            if cls in temp_0:
	                uv_x = min(max(int((bbox[0] + bbox[2])/2), 0), self.mapping_array.shape[2])
#	                uv_y = min(max(int(bbox[3]-120), 0), self.mapping_array.shape[1] - 1) 
	                uv_y = min(max(int((bbox[1] + bbox[3])/2-120), 0), self.mapping_array.shape[1] - 1)
	                uv = (uv_x, uv_y)
	                xyz = tuple(self.mapping_array[1][uv[1]][uv[0]]*1)
	            elif cls in temp_1:
	                uv_x = min(max(int((bbox[0] + bbox[2])/2), 0), self.mapping_array.shape[2])
	                uv_y = min(max(int((bbox[1] + bbox[3])/2-120), 0), self.mapping_array.shape[1] - 1)  
	                uv = (uv_x, uv_y)                    
	                xyz = tuple(self.mapping_array[1][uv[1]][uv[0]]*1)
	            elif cls in temp_2:
	                uv_x = min(max(int((bbox[0] + bbox[2])/2), 0), self.mapping_array.shape[2])
	                uv_y = min(max(int((bbox[1] + bbox[3])/2-120), 0), self.mapping_array.shape[1] - 1)
	                uv = (uv_x, uv_y) 
	                xyz = tuple(self.mapping_array[2][uv[1]][uv[0]]*1)
	            uv = (uv[0], uv[1]+120)
	            self.results_2d[ob_id][frame_id] = [uv, score, cls, c_f]	            
	            self.results_3d[ob_id][frame_id] = [xyz, score, cls, c_f]
	def edit_results(self):
		for i, ob_id in enumerate(self.results):
		    edit_flag = 0
		    for j, frame_id in enumerate(self.results[ob_id]):
		        if frame_id >= self.im_index:
			        if self.results[ob_id][frame_id][-1] >= self.continuous_frame_threshold \
                    and ob_id not in self.results_edit.keys():
			            bbox = self.results[ob_id][frame_id-1]
			            center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)  
			            _, xyz_3d = self.static_obj.c2dto3d(center[0], center[1])
			            kal_obj = Kalman_Filter(self.dts, 0)  
			            kal_g = kal_obj.ekf_chcv(xyz_3d[0], xyz_3d[1])
			            self.results_edit[ob_id] = {}
			            self.kal_dict[ob_id] = [kal_obj, kal_g]                        
			            edit_flag = 1
			        if edit_flag == 1 or ob_id in self.results_edit.keys():
			            bbox = self.results[ob_id][frame_id]
			            center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2) 
			            score = float(self.results[ob_id][frame_id][4])
			            cls = float(self.results[ob_id][frame_id][5])
			            c_f = float(self.results[ob_id][frame_id][6])
			            _, xyz_3d = self.static_obj.c2dto3d(center[0], center[1])
			            self.kal_dict[ob_id][0].set_xy(xyz_3d[0], xyz_3d[1])
			            x, y, psi, v = next(self.kal_dict[ob_id][1])
#			            self.results_edit[ob_id][frame_id] = [center[0], center[1]]
			            self.results_edit[ob_id][frame_id] = [bbox[0], bbox[1], bbox[2], bbox[3], score, cls, c_f]
	def step(self, blob):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
#		print("img_index:", self.im_index)
		for t in self.tracks:
			# add current position to last_pos list
			t.last_pos.append(t.pos.clone())

		#########################################
		# Look for new detections by mm or ctdet#
		#########################################
        # mm
#		all_result = self.obj_detect.mmapi_det(blob['img_ori'])
        # ctdet
#		t2 = time.time()
#        self.obj_detect.load_image(blob['img'])
		all_result = self.obj_detect.run(int(blob['idx']), blob['img_ori'])
#		t3 = time.time()
#		print('\n')
#		print('t3-t2:', t3-t2)
		car_list = []
		person_list = []
		for i in all_result:
			content = all_result[i]
			if i in self.cl_car_list:
				if content.shape[0] != 0:
				    content = torch.from_numpy(content)
				    cls = torch.ones([content.shape[0], 1])*i
				    content = torch.cat((content, cls), 1)
				    car_list.append(content)
			elif i in self.cl_person_list:
				if content.shape[0] != 0:
				    content = torch.from_numpy(content)
				    cls = torch.ones([content.shape[0], 1])*i
				    content = torch.cat((content, cls), 1)
				    person_list.append(content)                
        
		if len(person_list+car_list) > 0:
			res = torch.cat(person_list+car_list, 0).cuda()
			boxes = res[:,0:-2]
			scores = res[:,-2]
			clses = res[:,-1].view(-1, 1)
			inds = []
			keep = nms(boxes, scores, 0.8)
			scores_clone = scores.clone()
			keep_list = list(keep.cpu().numpy())
			keep_list_not = []
			for i in range(boxes.shape[0]):
				if i not in keep_list:
				    keep_list_not.append(i)
			for i in keep_list_not:           
			    scores_clone[i] += 1
			    keep_list_temp = list(nms(boxes, scores_clone, 0.8).cpu().numpy())    
			    add_score_index = list(set(keep_list).difference(set(keep_list_temp)))[0]
			    if (clses[add_score_index] in self.cl_car_truck_motor_list and clses[i] in self.cl_car_truck_motor_list \
				    and clses[add_score_index] != clses[i]) or (clses[add_score_index] in self.cl_motor_bike_person_list \
				    and clses[i] in self.cl_motor_bike_person_list and clses[add_score_index] != clses[i]):
				    scores[add_score_index] += scores[i]/2
			    scores_clone[i] -= 1            

                            
			for i, cl in enumerate(clses):
			    center = ((boxes[i][0].item()+boxes[i][2].item())/2, (boxes[i][1].item()+boxes[i][3].item())/2)
			    if cl in self.cl_car_truck_list:
			        dist = cv2.pointPolygonTest(self.region, center, False)
#			        dist = 1
			        if scores[i].item() > self.detection_car_thresh and dist > 0:
			            inds.append(i)
			    else:
			        dist = cv2.pointPolygonTest(self.region, center, False)
#			        dist = 1
			        if scores[i].item() > self.detection_person_thresh and dist > 0:
				        inds.append(i)
			inds.sort()
			inds = torch.cuda.LongTensor(inds)
			if inds.nelement() > 0:
			    det_pos = boxes[inds]
			    det_scores = scores[inds]
			    det_clses = clses[inds]
			else:
			    det_pos = torch.zeros(0).cuda()
			    det_scores = torch.zeros(0).cuda()  
			    det_clses = torch.zeros(0).cuda()
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()  
			det_clses = torch.zeros(0).cuda()            

		if self.do_reid:
			new_features = self.get_appearances(blob, det_pos)
        ##################
		# Predict tracks #
		##################
#		t10 = time.time()
		if len(self.tracks):
#			t4 = time.time()
			o_scores, o_clses, disappear_track, new_track = self.regress_tracks_mm(blob, det_pos, det_scores, det_clses)            
			if len(new_track) > 0:
                new_det_pos = det_pos[new_track]
                new_det_scores = det_scores[new_track]
                new_det_clses = det_clses[new_track]             
                
				new_det_pos, new_det_scores, new_det_clses, new_det_features = self.reid(blob, new_det_pos, new_det_scores, new_det_clses, new_features)	                
			if len(disappear_track) > 0:
				self.add(blob, new_det_pos, new_det_scores, new_det_clses, new_det_features)
#			t5 = time.time()
#			print('t5-t4:', t5-t4)
			if len(self.tracks):
				# create nms input
				# nms here if tracks overlap
				keep = nms(self.get_pos(), o_scores, self.regression_nms_thresh)
				keep = keep.sort()[0]
				self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])
#				t10 = time.time()
				if keep.nelement() > 0:
					if self.do_reid:
						new_features = self.get_appearances(blob)
						self.add_features(new_features)
            
#		t11 = time.time()
#		print('t11-t10', t11-t10)
		#####################
		# Create new tracks #
		#####################
#		if det_pos.nelement() > 0:
#			keep = nms(det_pos, det_scores, self.detection_nms_thresh)
#			keep = keep.sort()[0]
#			det_pos = det_pos[keep]
#			det_scores = det_scores[keep]
#			det_clses = det_clses[keep]
#
#			# check with every track in a single run (problem if tracks delete each other)
#			for t in self.tracks:
#				nms_track_pos = torch.cat([t.pos, det_pos])
#				nms_track_scores = torch.cat([torch.tensor([2.0]).to(det_scores.device), det_scores])
#				keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)
#				keep = keep.sort()[0]
#				keep = keep[torch.ge(keep, 1)] - 1
#
#				det_pos = det_pos[keep]
#				det_scores = det_scores[keep]
#				det_clses = det_clses[keep]
#				if keep.nelement() == 0:
#					break

#		if det_pos.nelement() > 0:
#			new_det_pos = det_pos
#			new_det_scores = det_scores
#			new_det_clses = det_clses
#			# try to reidentify tracks
#			new_det_pos, new_det_scores, new_det_clses, new_det_features = self.reid(blob, new_det_pos, new_det_scores, new_det_clses)	
#			# add new
#			if new_det_pos.nelement() > 0:
#				self.add(blob, new_det_pos, new_det_scores, new_det_clses, new_det_features)

		####################
		# Generate Results #
		####################
		clses = self.get_cls()
#		print('clses:', clses)
		for t in self.tracks:
#			pdb.set_trace()
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), \
                                                               np.array([t.cls.cpu()]), np.array([t.continuous_frame])])
#		self.edit_results()
#		self.mapping_results()
		for t in self.inactive_tracks:
			t.count_inactive += 1

		self.inactive_tracks = [
			t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
		]

		self.im_index += 1
#		self.last_image = blob['img'][0]

	def get_results(self):
		return self.results
	def get_results_edit(self):
		return self.results_edit
	def get_results_2d(self):
		return self.results_2d
	def get_results_3d(self):
		return self.results_3d
class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, cls, track_id, frame_id, features, inactive_patience, max_features_num, mm_steps, continuous_frame):
		self.id = track_id
		self.frame_id = frame_id
		self.pos = pos
		self.score = score
		self.cls = cls
		self.features = deque([features])
		self.smooth_feat = features
		self.alpha = 0.5    
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
		self.continuous_frame = continuous_frame
		self.last_v = torch.Tensor([])
		self.gt_id = None

	def has_positive_area(self):
		return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		if self.smooth_feat is None:
			self.smooth_feat = features
		else:
			self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * features        
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(list(self.features), dim=0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features, keepdim=True)
		return dist

	def test_features_cosine(self, test_features):
		"""Compares test_features to features of this Track object"""
		"""其中self.features是前面几帧当前目标的特征"""
#		if len(self.features) > 1:
#			features = torch.cat(list(self.features), dim=0)
#		else:
#			features = self.features[0]
#		features = features.mean(0, keepdim=True)
#		features = self.features[-1]
#		dist = torch.cosine_similarity(features.squeeze(), test_features.squeeze(), dim=0)
		dist = torch.cosine_similarity(self.smooth_feat.squeeze(), test_features.squeeze(), dim=0)
		if dist < 0:
			dist = torch.tensor(0.001).cuda()
		return dist
	def reset_last_pos(self):
		self.last_pos.clear()
		self.last_pos.append(self.pos.clone())
