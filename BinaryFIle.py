import struct

class BinaryFile:
	TYPE_INT, TYPE_FLOAT, TYPE_FLOATN, TYPE_INTN, TYPE_BITN, TYPE_FLOATMN, TYPE_INTMN, TYPE_BITMN, TYPE_STRING, TYPE_STRINGN, TYPE_ARRAY , TYPE_EOF =range(12)
	def __init__(self, filename, mode):
		self.f=open(filename, mode)
	def unpackStr(self):
		tid,n=struct.unpack('ii', self.f.read(8))
		assert(tid==self.TYPE_STRING)
		return ''.join(struct.unpack("<%dc" % n, self.f.read(n)))
	def unpackInt(self):
		tid,n=struct.unpack('ii', self.f.read(8))
		assert(tid==self.TYPE_INT)
		return n
	def unpackVec(self):
		tid,n=struct.unpack('ii', self.f.read(8))
		# print tid, self.TYPE_FLOATN
		assert(tid==self.TYPE_FLOATN)
		fmt="<%dd" % n
		return array(struct.unpack(fmt, self.f.read(8*n)))
	def close(self):
		self.f.close()
		def packInt(self,n):
			self.f.write(struct.pack('i', self.TYPE_INT))
			self.f.write(struct.pack('i', n))
		def packVec(self,v):
			self.f.write(struct.pack('i', self.TYPE_FLOATN))
			self.f.write(struct.pack('i', len(v)))
			param=["<%dd"%len(v)]+list(v)
			self.f.write(struct.pack(*param))


