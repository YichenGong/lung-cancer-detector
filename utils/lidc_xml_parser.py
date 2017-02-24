'''
File to parse the XMl file available in the diretory: ../data/lidc/XMLs/ 
'''

import xml.etree.ElementTree as ET
import os
import pickle

def extract_id(uid):
	return uid[uid.rfind('.')+1:]

class Point:
	def __init__(self, point_xml):
		self.x = self.y = float('nan')

		for child in point_xml:
			if child.tag.endswith("xCoord"):
				self.x = float(child.text)
			elif child.tag.endswith("yCoord"):
				self.y = float(child.text)
			else:
				print("Got a weird point", point_xml.text)

class NoduleROI:
	def __init__(self, roi_xml, isNodule=True):
		self.z = float('nan')
		self.edges = []
		self.image_uid = ''
		self.inclusion = False
		self.is_nodule = isNodule

		for child in roi_xml:
			if child.tag.endswith("imageZposition"):
				self.z = float(child.text)
			elif child.tag.endswith("imageSOP_UID"):
				self.image_uid = extract_id(child.text)
			elif child.tag.endswith("inclusion"):
				if child.text == "TRUE":
					self.inclusion = True
			elif child.tag.endswith("edgeMap") or child.tag.endswith("locus"):
				self.edges.append(Point(child))

	def get_edges(self):
		return self.edges

class Nodule:
	def __init__(self, nodule_xml_info):
		self._props = {}
		self._roi = []
		self._isNodule = True
		self._isBig = True
		self._id = ""

		self.__extract_info(nodule_xml_info)

	def __extract_characteristics(self, xml_info):
		for child in xml_info:
			if child.tag.endswith("subtlety"):
				self._props['subtle'] = int(child.text)
			elif child.tag.endswith("internalStructure"):
				self._props['structure'] = int(child.text)
			elif child.tag.endswith("calcification"):
				self._props['calcification'] = int(child.text)
			elif child.tag.endswith("sphericity"):
				self._props['sphericity'] = int(child.text)
			elif child.tag.endswith("margin"):
				self._props['margin'] = int(child.text)
			elif child.tag.endswith("lobulation"):
				self._props['lobulation'] = int(child.text)
			elif child.tag.endswith("spiculation"):
				self._props['spiculation'] = int(child.text)
			elif child.tag.endswith("texture"):
				self._props['texture'] = int(child.text)
			elif child.tag.endswith("malignancy"):
				self._props['malignancy'] = int(child.text)
			


	def __check_if_big_nodule(self):
		if len(self._roi) == 1:
			#Only one roi
			self._isBig = (len(self._roi[0].edges) > 1)

	def __construct_nodule(self, xml_info):
		self._isNodule = True

		for child in xml_info:
			if child.tag.endswith("noduleID"):
				self._id = child.text
			elif child.tag.endswith("characteristics"):
				self.__extract_characteristics(child)
			elif child.tag.endswith("roi"):
				self._roi.append(NoduleROI(child, isNodule=True))

		self.__check_if_big_nodule()

	def __construct_non_nodule(self, xml_info):
		self._isNodule = False

		for child in xml_info:
			if child.tag.endswith("nonNoduleID"):
				self._id = child.text

		self._roi.append(NoduleROI(xml_info, isNodule=False))

	def __extract_info(self, xml_info):
		#Check if the nodule is a real nodule
		if xml_info.tag.endswith("nonNodule"):
			self.__construct_non_nodule(xml_info)
		else:
			self.__construct_nodule(xml_info)

def parse_xml_reading(reading):
	nodules = []

	for child in reading:
		if child.tag.endswith("Nodule") or child.tag.endswith("unblindedRead"):
			nodules.append(Nodule(child))

	return nodules

def parse_xml_header(headers):
	imp_headers = {}
	for header in headers:
		if header.tag.endswith("TaskDescription"):
			if header.text == "Second unblinded read":
				imp_headers['type'] = 1
			elif header.text == "CXR read":
				imp_headers['type'] = 2
			else:
				#There are only two types of reads!
				print("Something different in header in header", header.text)
		elif header.tag.endswith("StudyInstanceUID"):
			#There are multiple kinds of UIDs available here
			#We'll only store Study Instance UID
			#If anything else is important, we'll update
			#the script accordingly
			# [TODO]: Check if this the correct UID to extract or a combination is required
			imp_headers['uid'] = extract_id(header.text)

	return imp_headers

def parse_xml(filepath):
	root = ET.parse(filepath).getroot()

	details = {'header': None, 'readings': []}

	for child in root:
		if child.tag.endswith("readingSession"):
			'''
			This is a reading from one of the four radiologists
			'''
			reading = parse_xml_reading(child)
			if len(reading) > 0:
				details['readings'].append(reading)
		else:
			'''
			It is a header
			Most of the things here are not important except UID
			'''
			details['header'] = parse_xml_header(child)

	# if details['header'] is None:
	# 	print("no header in XML file " + filepath)
	# if (len(details['readings']) == 0):
	# 	print("no readings in XML file " + filepath)

	return details

def load_xmls(dirPath):
	results = []

	for directory in os.listdir(dirPath):
		path = os.path.join(dirPath, directory)
		for xmlFile in os.listdir(path):
			results.append(parse_xml(os.path.join(path, xmlFile)))

	return results


if __name__ == "__main__":
	details = load_xmls('../data/lidc/XMLs')

	print("Number of details: ", len(details))