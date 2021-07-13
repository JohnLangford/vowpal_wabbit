import tensorboardX as tx
from tensorboardX.proto.graph_pb2 import GraphDef
from tensorboardX.proto.node_def_pb2 import NodeDef
from tensorboardX.proto import event_pb2

class VWtoTensorboard:
	"""The object of this class would be passed as a callback to DFtoVWtoTensorboard fit method to ensure writing of logs for Tensorboard"""

	def __init__(self, logdir):
		"""Construct a VWtoTensorboard object

		Parameters
		----------

		logdir : str
			A string specifying the log directory for Tensorboard logs

		Returns
		-------

		self : VWtoTensorboard
		"""
		self.file_writer = tx.SummaryWriter(logdir, flush_secs=30)   # creating file writer
		self.iteration = 0   # This would keep value of current iteration 

	def emit_learning_metrics(self, average_loss, since_last):
		"""This method for now logs the metrics given as arguments for Tensorboard visualization

		Parameters
		----------

		average_loss : float
				This is the average_loss value to be logged for Tensorboard
		since_last   : float
				This is the since_last value to be logged for Tensorboard

		Returns
		-------

		None
		"""
		self.file_writer.add_scalar('average_loss', average_loss, self.iteration)  # logging average_loss on each iteration
		self.file_writer.add_scalar('since_last', since_last, self.iteration)   # logging since_last on each iteration
		self.iteration += 1  # Now increment this as the incremented value is for next iteration

	def draw_reductions_graph(self, vw):
		"""This method draws vw model's reductions as graph for Tensorboard visualization

		Parameters
		----------

		vw    : A vowpalwabbit object
				This would be used to get the enabled reductions for making a graph

		Returns
		-------

		None
		"""
		nodes = []
		reductions = vw.get_enabled_reductions()

		for ind, layer in enumerate(reductions):
			if ind == 0:
				nodes.append(NodeDef(name=layer, op='op'))   # First node does not have any input edge
			else:
				nodes.append(NodeDef(name=layer, op='op', input=[reductions[ind-1]]))

		g = GraphDef(node=nodes)
		event = event_pb2.Event(graph_def=g.SerializeToString())
		self.file_writer._get_file_writer().add_event(event)


class DFtoVWtoTensorboard:
	"""This class provides capability for learning possible on DFtoVW object, provides capability to 
	print metrics (average_loss, since_last, label, prediction, num_features) and also provides support 
	to log these metrics for Tensorboard Visualization"""

	def __init__(self, vw_formatted_data, vw):
		"""Construct a DFtoVWtoTensorboard object

		Parameters
		----------

		vw_formatted_data : A DFtoVW object
				This would be the list of examples provided either by DFtoVW.convert_df or a user making custom examples
		vw		 : A vowpalwabbit object
 				This would be used for learning using the algorithm specified in the object, and also to compute a few metrics

		Returns
		-------

		self : DFtoVWtoTensorboard
		"""
		self.vw = vw  # As this needs to be remembered
		self.vw_formatted_data = vw_formatted_data


	#------------------------------------------------------------------------------------------------		
	def fit(self, vw_to_tensorboard=None):
		"""Learns on the relevant examples and can also log metrics for tensorboard visualization

		Parameters
		----------

		vw_to_tensorboard  : VWtoTensorboard object 
					Default value is None, this parameter is used to control the logging of metrics for Tensorboard visualization
					If value is VWtoTensorboard object : metrics are computed and logged for Tensorboard visualization
					If value is None : metrics are not computed and not logged for Tensorboard visualization

		Returns
		-------

		None
		"""
		for iteration, vw_format in enumerate(self.vw_formatted_data):
			self.vw.learn(vw_format, vw_to_tensorboard)         # learn on example

