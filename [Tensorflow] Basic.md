# Overview / High-level Concepts:
## Computation
1. Represent computation as a graph, called **ComputationalGraph**
2. Execute the graph in a **Session**: 
 3. A **Session** instance will place the nodes in **ComputationGraph** into the **Device**s (such as GPU or CPU) node. 
 4.  A **session** then calls the corresponding implementations for execution (such as *numpy.ndarray* or C/C++ *tensorflow::Tensor* instance) 

## Data
 1. Represent data as **Tensor**
 2. Maintain computation result in **Variable** (mutable **Tensor**)
 3. using feed and fetch to get the data in and out through a **Session** and the computation result based on the **ComputationalGraph** can be acquired through fetching. 
 
 # Low Level API:
1.  *tf.Graph*: Represet a computation graph which consists the components, *tf.constant*, *tf.Varible* and *tf.Operation*. 
2. *tf.constant*: A node in  **ComputationalGraph**  without any inputs (itself can be viewed as a minimal *tf.Graph* and also can be embed in other larger graph). It has the following characteristics:       
     1.  Immutable: Once the value passing into the *tf.Constant* constructor, it cannot be changed      
      2.  Initial data value must pass when calling constructor 
      3.  *tf.Session()* should run firstly and then  pass the *tf.constant*  instance to *tf.Session().run* method in order to get the value held by *tf.constant* instance. 
 3.  *tf.Variable*: stateful tensor variables
	 1.  is Mutable and whose value can be changed through *tf.Variable().assign* method
	 2.  Frequently used parameters of the constructor 
	      1.  **variable_def**  and **initial_value** are mutual exclusive:  if **variable_def** is given, then **initial_value** must be None; if **variable_def** is not given, then **initial_value** must be given (either **variable_def** or **initial_value** can be present) where 
	           1.   **initial_value**=\<data used to construct type and shape\> and default value is None
	           2.   **variable_def**=\<protobuf instance or None\> and default value is None. 
	      4. **trainable**:  The default value is True. Whe setting to False, it won't be put into the trainable collection, see [below](#tf-variable). 
		 3. **validate_shape**: True
 4.  *tf.Operation*:
       1. Utility Ops  including for *tf.Variable*.assign()
	   2. Arithmetic Ops: including reduce type operation (including reduce_mean etc) 
	   3. Initialization Ops: including global initializer of globally shared variables and local initializer
	   4. Neural network specific Ops: see *tf.nn*
            
<h2 id=tf-variable> More about tf.Variable </h2>

### Variable collections: 
1. Global:  In this collection, variables are shared across different devices. One can use **tf.GraphKeys.GLOBAL_VARIABLES** as key to fetch. 
2. Trainable:   In this collection, variables will be computed graident. One can use **tf.GraphKeys.TRAINABLE_VARIABLES** as key to fetch and store. 
3. Not Trainable:   To do so, one can add the variables to **tf.GraphKeys.LOCAL_VARIABLES** collection. 

Other collections used in *tf.estimator*:
ops is *tensorflow.python.framework.ops* package
- ops.GraphKeys.LOSSES
- ops.GraphKeys.SUMMARIES
- ops.GraphKeys.SAVERS
The following code is to add variable to non-trainable collection
```python
code here
```
### Name scope :
of *tf.Variable*
5. initialize *tf.Varaible*: initializer: due to Variable might not hold valid data (initial_value could be None) and needs to run through its initializer (an instance of tensorflow.op.Operation). If not executing \<variable\>.initializer.run() will throw FailedPreconditionError
6. global_variables_initializer using tf.global_variables() return global variables which are shared across different machines given Graph

More about *tf.Graph*
7.   description of tf.Graph:
	1.  Each thread can only have one global tf.Graph instance but nested graph scope can be created within the global Graph
	2.  getter: using tf.get_default_graph() to get the currently owned global tf.Graph instance for that thread or the innermost graph created through nested scope via with Graph.as_default() context scope
	3. . setter: using with g.as_default() context scope to set the newly created tf.Graph instance, g as default graph within the context

8.  namescope of tf.Graph:
	1.  More about g.as_default(): used when multiple graphs are needed to be created
	2.  default behavior of default graph: tensorflow will use one global graph per thread and any created operations without explicitly attaching to a specific graph will be added to the global default graph (any operations created by using tf.<operation> without being in any g.as_default() with context)      
            3.  calling g.name_scope() to create hierarchical name scopes
                
9.  attributes of tf.Graph:
        1.  collections: user can group some variables or operations by certain condition and giving meaningful identifiable name in collection (using g.get_all_collection_key())
         2.  building_function:
                
	3.  reference: [Graph class](https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph)
        
10.  tf.Sessions: evaluation the ComputationalGraph with a C++ backend (tf.Tensor often has their own eval method which also take a feed_dict)
    
11.  1.  launch session default device (should be CPU) with interactive session such in IPython
        
    2.  1.  tensorflow.InteractiveSession()
            
    3.  launch session default device (should be CPU) by simply calling session.run (off-line, not interactive)
        
    4.  launch session with specific device (advanced, required knowledge about GPU, [later](https://www.tensorflow.org/how_tos/using_gpu/index), #FUTURE)
        
    5.  launch session with distributed cluster (advanced, required knowledge about process communication, [later](https://www.tensorflow.org/how_tos/distributed/), #FUTURE)
    
# Store and  Restore
## Save 
1. Save for training later
2. Save for serving: building a *tf.saved_model*

### Low level API: *tf.train.Saver*
1. Low level saver such as **tf.train.Saver** has the control to save 


# Debugger:
    
6.  1.  tensorflow: [https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html)
        
 # High Level API:
12.  Keras is included in tensorflow as *tf.keras* : 
 2.  3rd party [TFLearn](http://tflearn.org/)(http://tflearn.org/)
        
# Terminology:
 1. Sesison: a C/C++ backend which would prepare an execution environment for the built **ComputationalGraph**
 2. **Tensor**:  typed multi-dimensional array            
5. **Variable** : a stateful object which can hold data buffer and will sync with other nodes under its scope with its own run method            

7.  Basic Usage for Tensor:
	 1.  Tensor Transformations:
		 - shapes and shaping:
 4. Training:
		- tensorflow.train
            


    2.  1.  pdb cannot access C/C++ code while gdb cannot access high-level graph structure in python (paper downloaded, tfdebugger.pdf)

> Written with [StackEdit](https://stackedit.io/).

                
            2. 
                    
        7.  reference:
            

  
5.  Feed and Fetch:
    
6.  1.  fetch: execute session.run(<single or a list of Variable>)
        
    2.  feed:
        
    3.  1.  tensorflow.placeholder: when evaluating in the session with Session.run method providing a dict whose name as the placeholder
            
        2.  preallocate method: using tensorflow.constant and tensorflow.Variable
