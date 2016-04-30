/**
* library_cuda.js
* Licence : https://github.com/wolfviking0/webcl-translator/blob/master/LICENSE
*
* Created by Anthony Liot.
* Copyright (c) 2013 Anthony Liot. All rights reserved.
*
* @module LibraryCUDA
*/
var LibraryCUDA = {  
  /**
  * Contains some method for the LibraryCUDA Module
  *
  * @class CU
  * @static
  */
  $CU: {
    cuda_types: { 0 : 'i32', 1 : 'float' , 2 : 'i32'},
    cuda_init: 0,
    cuda_from_type: 4/*DEVICE_TYPE_GPU*/,
    cuda_context: 0,
    cuda_device: 0,
    cuda_command_queue: 0,
    cuda_user_event: 0,
    cuda_digits: [1,2,3,4,5,6,7,8,9,0],
    cuda_objects: {},
    // Events stuffs
    cuda_profile_event: false,
    cuda_events: [],

    // Errors
    cuda_errors: [],

    /**
     * Initialize all the attribute of the CU Class
     * 
     * @method init
     * @return {int} CU.cuda_init(0 / 1)
     */
    init: function() {
      if (CU.cuda_init == 0) {
        
        console.log('%c CU2WebCL-Translator V1.0 ! ', 'background: #222; color: #bada55');

        if (webcl == undefined) {
        
          alert("Unfortunately your system does not support WebCL. " +
          "Make sure that you have WebKit Samsung or Firefox Nokia plugin");

          console.error("Unfortunately your system does not support WebCL.\n");
          console.error("Make sure that you have WebKit Samsung or Firefox Nokia plugin\n");  
        } else {
#if CU_GRAB_TRACE
          CU.cudaBeginStackTrace("CU.init",[]);
#endif
          // Just for call cuda 
          cuda = webcl;

          // Get the first GPU device and create a context
#if CU_GRAB_TRACE
          CU.cudaCallStackTrace( cuda+".getPlatforms",[]);
#endif
          var _platforms  = cuda.getPlatforms();
#if CU_GRAB_TRACE
          CU.cudaCallStackTrace( _platforms[0]+".getDevices",[CU.cuda_from_type]);
#endif          
          var _devices    = _platforms[0].getDevices(CU.cuda_from_type);
#if CU_GRAB_TRACE
          CU.cudaCallStackTrace( cuda+".createContext",[_devices[0]]);
#endif          
          var _context    = cuda.createContext(_devices[0]);  
#if CU_GRAB_TRACE
          CU.cudaCallStackTrace( _context+".createCommandQueue",[_devices[0],cuda.QUEUE_PROFILING_ENABLE]);
#endif              
          var _command    = _context.createCommandQueue(_devices[0],cuda.QUEUE_PROFILING_ENABLE);  

          // Grab Id
          CU.cuda_context       = CU.udid(_context);
          CU.cuda_device        = CU.udid(_devices[0]);
          CU.cuda_command_queue = CU.udid(_command);

          // Init
          CU.cuda_init = 1;

#if CU_GRAB_TRACE
          CU.cudaEndStackTrace([1],"","");
#endif          
        }

      }

      return CU.cuda_init;
    },

    /**
     * Description
     * @method udid
     * @param {} obj
     * @return _id
     */
    udid: function (obj) {    
      var _id;

      if (obj !== undefined) {

        if ( obj.hasOwnProperty('udid') ) {
         _id = obj.udid;

         if (_id !== undefined) {
           return _id;
         }
        }
      }

      var _uuid = [];

      _uuid[0] = CU.cuda_digits[0 | Math.random()*CU.cuda_digits.length-1]; // First digit of udid can't be 0
      for (var i = 1; i < 6; i++) _uuid[i] = CU.cuda_digits[0 | Math.random()*CU.cuda_digits.length];

      _id = _uuid.join('');

#if CU_DEBUG
      if (_id in CU.cuda_objects) {
        console.error("/!\\ **********************");
        console.error("/!\\ UDID not unique !!!!!!");
        console.error("/!\\ **********************");        
      }
#endif
    
      // /!\ Call udid when you add inside cl_objects if you pass object in parameter
      if (obj !== undefined) {
        Object.defineProperty(obj, "udid", { value : _id,writable : false });
        CU.cuda_objects[_id]=obj;
      }

      return _id;      
    },

    /**
     * Description
     * @method catchError
     * @param {} e
     * @return _error
     */
    catchError: function(e) {
      console.error(e);
      var _error = -1;

      if (e instanceof WebCLException) {
        var _str=e.message;
        var _n=_str.lastIndexOf(" ");
        _error = _str.substr(_n+1,_str.length-_n-1);
      }
      
      return _error;
    },

    /**
     * Description
     * @method convertCudaKernelToOpenCL
     * @param {} kernel
     * @return _kernelConverted
     */
    convertCudaKernelToOpenCL: function(kernel, name) {

      // Remove all comments ...
      var _kernelConverted  = kernel.replace(/(?:((["'])(?:(?:\\\\)|\\\2|(?!\\\2)\\|(?!\2).|[\n\r])*\2)|(\/\*(?:(?!\*\/).|[\n\r])*\*\/)|(\/\/[^\n\r]*(?:[\n\r]+|$))|((?:=|:)\s*(?:\/(?:(?:(?!\\*\/).)|\\\\|\\\/|[^\\]\[(?:\\\\|\\\]|[^]])+\])+\/))|((?:\/(?:(?:(?!\\*\/).)|\\\\|\\\/|[^\\]\[(?:\\\\|\\\]|[^]])+\])+\/)[gimy]?\.(?:exec|test|match|search|replace|split)\()|(\.(?:exec|test|match|search|replace|split)\((?:\/(?:(?:(?!\\*\/).)|\\\\|\\\/|[^\\]\[(?:\\\\|\\\]|[^]])+\])+\/))|(<!--(?:(?!-->).)*-->))/g, "");
      
      // Remove all char \n \r \t ...
      _kernelConverted = _kernelConverted.replace(/\n/g, " ");
      _kernelConverted = _kernelConverted.replace(/\r/g, " ");

      // Remove all the multispace
      _kernelConverted = _kernelConverted.replace(/\s{2,}/g, " ");
      _kernelConverted = _kernelConverted.replace(/template[A-Za-z0-9_\s]+\<([^)]+)\>/g, "");

      // Convert common CUDA equivalent to OpenCL
      _kernelConverted = _kernelConverted.replace("__global__","__kernel");
      _kernelConverted = _kernelConverted.replace(/__shared__/g,"__local");
      _kernelConverted = _kernelConverted.replace(/__constant__/g,"__constant");

      _kernelConverted = _kernelConverted.replace(/__syncthreads\(\)/g,"barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)");

      _kernelConverted = _kernelConverted.replace(/blockDim.x/g,"get_local_size(0)");
      _kernelConverted = _kernelConverted.replace(/blockDim.y/g,"get_local_size(1)");
      _kernelConverted = _kernelConverted.replace(/blockDim.z/g,"get_local_size(2)");

      _kernelConverted = _kernelConverted.replace(/blockIdx.x/g,"get_group_id(0)");
      _kernelConverted = _kernelConverted.replace(/blockIdx.y/g,"get_group_id(1)");
      _kernelConverted = _kernelConverted.replace(/blockIdx.z/g,"get_group_id(2)");

      _kernelConverted = _kernelConverted.replace(/threadIdx.x/g,"get_local_id(0)");
      _kernelConverted = _kernelConverted.replace(/threadIdx.y/g,"get_local_id(1)");
      _kernelConverted = _kernelConverted.replace(/threadIdx.z/g,"get_local_id(2)");

      // Add space qualifier : kernel pointer arguments must have a global, local, or constant address 
      var _matches = _kernelConverted.match(/__kernel[A-Za-z0-9_\s]+\(([^)]+)\)/g);
      
      for (var i = 0; i < _matches.length; i++) {

        var _braceOpen  = _matches[i].indexOf("(");
        var _braceClose = _matches[i].indexOf(")");
        var _parameter  = _matches[i].substr(_braceOpen + 1,_braceClose - _braceOpen - 1);

        var _paramList = _parameter.split(",");
        for (var j = 0; j < _paramList.length; j++) {
          if ( (_paramList[j].indexOf("*") != -1) && (_paramList[j].indexOf("__local") == -1) && (_paramList[j].indexOf("__constant") == -1)) {
            // Pointer 
            _kernelConverted = _kernelConverted.replace(_paramList[j],"__global "+_paramList[j]);
          }
        }
      }

      return _kernelConverted;
    },

#if CU_GRAB_TRACE     
    stack_trace_offset: -1,
    stack_trace_complete: "// Javascript cuda Stack Trace\n\n",
    stack_trace: "",

    /**
     * Description
     * @method cudaBeginStackTrace
     * @param {} name
     * @param {} parameter
     * @return 
     */
    cudaBeginStackTrace: function(name,parameter) {
      if (CU.stack_trace_offset == -1) {
        CU.stack_trace_offset = "";
      } else {
        CU.stack_trace_offset += "\t";
      }

      CU.stack_trace += "\n" + CU.stack_trace_offset + name + "("

      CU.cudaCallParameterStackTrace(parameter);

      CU.stack_trace += ")\n";
    },
                                                              
    /**
     * Description
     * @method cudaCallStackTrace
     * @param {} name
     * @param {} parameter
     * @return 
     */
    cudaCallStackTrace: function(name,parameter) {
      CU.stack_trace += CU.stack_trace_offset + "\t->" + name + "("

      CU.cudaCallParameterStackTrace(parameter);

      CU.stack_trace += ")\n";
    },

    /**
     * Description
     * @method cudaCallParameterStackTrace
     * @param {} parameter
     * @return 
     */
    cudaCallParameterStackTrace: function(parameter) {
      
      for (var i = 0; i < parameter.length - 1 ; i++) {
        if ( 
          (parameter[i] instanceof Uint8Array)    ||
          (parameter[i] instanceof Uint16Array)   ||
          (parameter[i] instanceof Uint32Array)   ||
          (parameter[i] instanceof Int8Array)     ||
          (parameter[i] instanceof Int16Array)    ||
          (parameter[i] instanceof Int32Array)    ||
          (parameter[i] instanceof Float32Array)  ||          
          (parameter[i] instanceof ArrayBuffer)   ||            
          (parameter[i] instanceof Array)){ 

          CU.stack_trace += "[";  
          for (var j = 0; j < Math.min(25,parameter[i].length - 1) ; j++) {
            CU.stack_trace += parameter[i][j] + ",";
          }
          if (parameter[i].length > 25) {
            CU.stack_trace += " ... ,";
          }
          if (parameter[i].length >= 1) {
            CU.stack_trace += parameter[i][parameter[i].length - 1];
          }
          CU.stack_trace += "],";
        } else {
          CU.stack_trace += parameter[i] + ",";  
        }
      }

      if (parameter.length >= 1) {
        if ( 
          (parameter[parameter.length - 1] instanceof Uint8Array)    ||
          (parameter[parameter.length - 1] instanceof Uint16Array)   ||
          (parameter[parameter.length - 1] instanceof Uint32Array)   ||
          (parameter[parameter.length - 1] instanceof Int8Array)     ||
          (parameter[parameter.length - 1] instanceof Int16Array)    ||
          (parameter[parameter.length - 1] instanceof Int32Array)    ||
          (parameter[parameter.length - 1] instanceof Float32Array)  ||          
          (parameter[parameter.length - 1] instanceof ArrayBuffer)   ||  
          (parameter[parameter.length - 1] instanceof Array)){ 

          CU.stack_trace += "[";  
          for (var j = 0; j < Math.min(25,parameter[parameter.length - 1].length - 1) ; j++) {
            CU.stack_trace += parameter[parameter.length - 1][j] + ",";
          }
          if (parameter[parameter.length - 1].length > 25) {
            CU.stack_trace += " ... ,";
          }
          if (parameter[parameter.length - 1].length >= 1) {
            CU.stack_trace += parameter[parameter.length - 1][parameter[parameter.length - 1].length - 1];
          }
          CU.stack_trace += "]";
        } else {
          CU.stack_trace += parameter[parameter.length - 1]; 
        }
      }
    },

    /**
     * Description
     * @method cudaEndStackTrace
     * @param {} result
     * @param {} message
     * @param {} exception
     * @return 
     */
    cudaEndStackTrace: function(result,message,exception) {
      CU.stack_trace += CU.stack_trace_offset + "\t\t=>Result (" + result[0];
      if (result.length >= 2) {
        CU.stack_trace += " : ";
      }

      for (var i = 1; i < result.length - 1 ; i++) {
        CU.stack_trace += ( result[i] == 0 ? '0' : {{{ makeGetValue('result[i]', '0', 'i32') }}} ) + " - ";
      }

      if (result.length >= 2) {
        CU.stack_trace +=  ( result[result.length - 1] == 0 ? '0' : {{{ makeGetValue('result[result.length - 1]', '0', 'i32') }}} );
      }

      CU.stack_trace += ") - Message (" + message + ") - Exception (" + exception + ")\n";

#if CU_PRINT_TRACE
      console.info(CU.stack_trace);
      //alert(CU.stack_trace); // Useful for step by step debugging
#endif   
      CU.stack_trace_complete += CU.stack_trace;
      CU.stack_trace = "";

      if (CU.stack_trace_offset == "") {
        CU.stack_trace_offset = -1;
      } else {
        CU.stack_trace_offset = CU.stack_trace_offset.substr(0,CU.stack_trace_offset.length-1);
      }
    },
#endif
  },

  /**
   * Method for acces to the stack trace debugger
   *
   * @method cudaPrintStackTrace
   * @param {string} param_value Receive the stack trace String
   * @param {int} param_value_size Size of the stack trace
   * @return {int} cudaSuccess(0)
   */
  cudaPrintStackTrace: function(param_value,param_value_size) {
#if CU_GRAB_TRACE
    var _size = {{{ makeGetValue('param_value_size', '0', 'i32') }}} ;
    
    if (_size == 0) {
      {{{ makeSetValue('param_value_size', '0', 'CU.stack_trace_complete.length + 1', 'i32') }}} /* Size of char stack */;
    } else {
      writeStringToMemory(CU.stack_trace_complete, param_value);
    }
#else
    {{{ makeSetValue('param_value_size', '0', '0', 'i32') }}}
#endif    
    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaEventCreate
   * @param {} event
   * @param {} flags
   * @return cudaSuccess(0), cudaErrorInitializationError(3), cudaErrorInvalidValue(11), cudaErrorLaunchFailure(4), cudaErrorMemoryAllocation(2)
   */
  cudaEventCreate: function(event, flags) {

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaEventCreate",[event, flags]);
#endif

    {{{ makeSetValue('event', '0', '1', 'i32') }}}

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif
  
    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaEventRecord
   * @param {} event
   * @param {} stream   
   * @return cudaSuccess(0), cudaErrorInvalidValue(11), cudaErrorInitializationError(3), cudaErrorInvalidResourceHandle(33), cudaErrorLaunchFailure(4)   
   */
  cudaEventRecord: function(event, stream) {

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaEventRecord",[event, stream]);
#endif

    CU.cuda_profile_event = true;

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0,event],"","");
#endif
  
    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaEventSynchronize (equivalent clWaitForEvents)
   * @return cudaSuccess(0), cudaErrorInitializationError(3), cudaErrorInvalidValue(11), cudaErrorInvalidResourceHandle(33), cudaErrorLaunchFailure(4)
   */
  cudaEventSynchronize: function(event) {

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaEventSynchronize",[event]);
#endif

    if (CU.cuda_events.length == 0) {
#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([11],"Cuda event is null !!!","");
#endif    
      return 11 /* cudaErrorInvalidValue */; 
    }

    try {

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(""+cuda+".waitForEvents",[CU.cuda_events]);
#endif      
      cuda.waitForEvents(CU.cuda_events);

    } catch (e) {

      var _error = CU.catchError(e);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([_error],"",e.message);
#endif
      return _error;

    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaEventElapsedTime
   * @param {} ms - Time between start and end in ms
   * @param {} start - Starting event
   * @param {} end - Ending event
   * @return cudaSuccess(0), cudaErrorNotReady(34), cudaErrorInvalidValue(11), cudaErrorInitializationError(3), cudaErrorInvalidResourceHandle(33), cudaErrorLaunchFailure(4)
   */
  cudaEventElapsedTime: function(ms,start,end) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaEventElapsedTime",[]);
#endif

    try { 
      
      var _ms = 0;

      for (var i = 0; i < CU.cuda_events.length; i++) {
#if CU_GRAB_TRACE
        CU.cudaCallStackTrace(""+CU.cuda_events[i]+".getProfilingInfo",[cuda.PROFILING_COMMAND_START]);
#endif

        var _info_start = CU.cuda_events[i].getProfilingInfo(cuda.PROFILING_COMMAND_START);

#if CU_GRAB_TRACE
        CU.cudaCallStackTrace(""+CU.cuda_events[i]+".getProfilingInfo",[cuda.PROFILING_COMMAND_END]);
#endif

        var _info_end = CU.cuda_events[i].getProfilingInfo(cuda.PROFILING_COMMAND_END);

        _ms += _info_end - _info_start;  

#if CU_GRAB_TRACE
        CU.cudaCallStackTrace(""+CU.cuda_events[i]+".release",[]);
#endif
        CU.cuda_events[i].release();
      }

      CU.cuda_events.length = 0;

      _ms *= 0.000001;

      {{{ makeSetValue('ms', '0', '_ms', 'float') }}}

    } catch (e) {
      var _error = CU.catchError(e);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([34],"",e.message);
#endif
      return 34 /* cudaErrorNotReady */;
    }    

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0,ms],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaSetDevice
   * @param {} device  - Device on which the active host thread should execute the device code.
   * @return cudaSuccess(0), cudaErrorInvalidDevice(9), cudaErrorDeviceAlreadyInUse()
   */
  cudaSetDevice: function(device_type) {

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaSetDevice",[device_type]);
#endif

    // 0  : DEVICE_TYPE_GPU
    // 1  : DEVICE_TYPE_CPU
    // 2  : DEVICE_TYPE_ACCELERATOR
    // 3  : DEVICE_TYPE_ALL
    // ...: DEVICE_TYPE_DEFAULT
#if CU_DEBUG
    var _device = device_type==0?"DEVICE_TYPE_GPU":device_type==1?"DEVICE_TYPE_CPU":device_type==2?"DEVICE_TYPE_ACCELERATOR":device_type==3?"DEVICE_TYPE_ALL":"DEVICE_TYPE_DEFAULT";
    console.info("cudaSetDevice convert Cuda Device("+device_type+") to WebCL Device("+ _device +")")
#endif
    switch (device_type) {
      case 0:
        CU.cuda_from_type = 0x4;
        break;
      case 1:
        CU.cuda_from_type = 0x2;
        break;
      case 2:
        CU.cuda_from_type = 0x8;
        break;
      case 3:
        CU.cuda_from_type = 0xFFFFFFFF;
        break;
      default:
        CU.cuda_from_type = 0x1;
    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaGetDevice
   * @param {} device - Returns the device on which the active host thread executes the device code.
   * @return cudaSuccess(0), cudaErrorUnknown(30)
   */
  cudaGetDevice: function(device) {
    var _initialize = CU.init();

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaGetDevice",[device]);
#endif

    // Init webcl variable if necessary
    if (_initialize == 0) {
#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([2],"webcl is not found !!!!","");
#endif
      return 30; /* cudaErrorUnknown */
    }

    {{{ makeSetValue('device', '0', 'CU.cuda_from_type', 'i32') }}};

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0,device],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaGetDeviceProperties
   * @param {} prop  - Properties for the specified device
   * @param {} device  - Device number to get properties for
   * @return cudaSuccess(0), cudaErrorInvalidDevice(10)
   */
  cudaGetDeviceProperties: function(prop, device) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaGetDeviceProperties",[prop, device]);
#endif
  
    var _device = CU.cuda_objects[CU.cuda_device];

    if (CU.cuda_from_type != device) {
      return 10; /* cudaErrorInvalidDevice */
    }

    var _info = "";
    var _type = _device.getInfo(cuda.DEVICE_TYPE);
    switch (_type) {
      case cuda.DEVICE_TYPE_CPU:
        _info = "WEBCL_DEVICE_CPU";
      break;
      case cuda.DEVICE_TYPE_GPU:
        _info = "WEBCL_DEVICE_GPU";
      break;
      case cuda.DEVICE_TYPE_ACCELERATOR:
        _info = "WEBCL_DEVICE_ACCELERATOR";
      break;
      case cuda.DEVICE_TYPE_DEFAULT:
        _info = "WEBCL_DEVICE_DEFAULT";
      break;
    }
    
    //char name[256];
    writeAsciiToMemory(_info,prop);

    // int major;
    {{{ makeSetValue('prop', '312', '1', 'i32') }}};

    // int minor;
    {{{ makeSetValue('prop', '316', '0', 'i32') }}};

    /*
    struct cudaDeviceProp {
        char name[256]; 
        size_t totalGlobalMem; 264
        size_t sharedMemPerBlock; 272
        int regsPerBlock; 276
        int warpSize; 280
        size_t memPitch; 288
        int maxThreadsPerBlock; 292
        int maxThreadsDim[3]; 304
        int maxGridSize[3]; 316
        int clockRate; 320
        size_t totalConstMem; 328
        int major; 332
        int minor; 336
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
    }
    */

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0,prop],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaDeviceSynchronize (clFinish)
   * @return cudaSuccess(0), cudaErrorUnknown(30)
   */
  cudaDeviceSynchronize: function() {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaDeviceSynchronize",[]);
#endif

    try {

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(""+CU.cuda_objects[CU.cuda_command_queue]+".finish",[]);
#endif      
      CU.cuda_objects[CU.cuda_command_queue].finish();

    } catch (e) {

      var _error = CU.catchError(e);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([30],"",e.message);
#endif

      return 30; /* cudaErrorUnknown */

    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaMalloc (clCreateBuffer)
   * @param {} devPtr - Pointer to allocated device memory
   * @param {} size   - Requested allocation size in bytes
   * @return cudaSuccess(0), cudaErrorMemoryAllocation(2)
   */
  cudaMalloc: function(devPtr, size) {
    var _initialize = CU.init();

#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaMalloc",[devPtr, size]);
#endif

    // Init webcl variable if necessary
    if (_initialize == 0) {
#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([2],"webcl is not found !!!!","");
#endif
      return 2; /* cudaErrorMemoryAllocation */
    }
    
    var _id = null;
    var _buffer = null;
    var _flags = cuda.MEM_READ_WRITE

    try {

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace( CU.cuda_objects[CU.cuda_context]+".createBuffer",[_flags,size]);
#endif   

      _buffer = CU.cuda_objects[CU.cuda_context].createBuffer(_flags,size);

      _id = CU.udid(_buffer);

      {{{ makeSetValue('devPtr', '0', '_id', 'i32') }}};

    } catch (e) {

      var _error = CU.catchError(e);
    
#if CL_GRAB_TRACE
      CU.cudaEndStackTrace([2],"",e.message);
#endif
      return 2; /* cudaErrorMemoryAllocation */
    }
    

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0,devPtr],"","");
#endif

    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaMemcpy (clEnqueueWriteBuffer or clEnqueueReadBuffer)
   * @param {} dst   - Destination memory address
   * @param {} src   - Source memory address
   * @param {} count - Size in bytes to copy
   * @param {} kind  - Type of transfer
   * @return cudaSuccess(0), cudaErrorInvalidValue(11), cudaErrorInvalidDevicePointer(17), cudaErrorInvalidMemcpyDirection(21)
   */
  cudaMemcpy: function(dst,src,count,kind) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaMemcpy",[dst,src,count,kind]);
#endif

    try {

      if (kind == 1 /*cudaMemcpyHostToDevice <-> clEnqueueWriteBuffer*/) {

        var _host_ptr = {{{ makeHEAPView('F32','src','src+count') }}};

#if CU_GRAB_TRACE
        CU.cudaCallStackTrace(""+CU.cuda_objects[CU.cuda_command_queue]+".enqueueWriteBuffer",[CU.cuda_objects[dst],true,0,count,_host_ptr,[]]);
#endif          
        
        CU.cuda_objects[CU.cuda_command_queue].enqueueWriteBuffer(CU.cuda_objects[dst],true,0,count,_host_ptr,[]);    

      } else /*cudaMemcpyDeviceToHost <-> clEnqueueReadBuffer*/ {

        var _host_ptr = {{{ makeHEAPView('F32','dst','dst+count') }}};
        
        CU.cuda_objects[CU.cuda_command_queue].enqueueReadBuffer(CU.cuda_objects[src],true,0,count,_host_ptr,[]);    

#if CU_GRAB_TRACE
        CU.cudaCallStackTrace(""+CU.cuda_objects[CU.cuda_command_queue]+".enqueueReadBuffer",[CU.cuda_objects[src],true,0,count,_host_ptr,[]]);
#endif          

      }

    } catch (e) {

      var _error = CU.catchError(e);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([17],"",e.message);
#endif

      return 17; /* cudaErrorInvalidDevicePointer */

    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif
    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaFree (clReleaseMemObject)
   * @param {} devPtr   - Device pointer to memory to free
   * @return cudaSuccess(0), cudaErrorInitializationError(3), cudaErrorInvalidDevicePointer(17)
   */
  cudaFree: function(devPtr) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaFree",[devPtr]);
#endif

    try {

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(CU.cuda_objects[devPtr]+".release",[]);
#endif        
      CU.cuda_objects[devPtr].release();
      delete CU.cuda_objects[devPtr];  

    } catch (e) {

      var _error = CU.catchError(e);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([17],"",e.message);
#endif

      return 17; /* cudaErrorInvalidDevicePointer */

    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif
    return 0; /* cudaSuccess */
  },

  /**
   * Description
   * @method cudaDeviceReset (no direct equivalent)
   * @return cudaSuccess(0)
   */
  cudaDeviceReset: function() {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaDeviceReset",[]);
#endif

#if CU_GRAB_TRACE
    CU.cudaCallStackTrace(CU.cuda_objects[CU.cuda_context]+".release",[]);
#endif        
    
    //CU.cuda_objects[CU.cuda_context].releaseAll();
    //delete CU.cuda_objects[CU.cuda_context];  

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif
    return 0; /* cudaSuccess */
  },

 /**
   * Description
   * @method cudaGetLastError (no direct equivalent)
   * @return cudaSuccess(0), ...
   */
  cudaGetLastError: function() {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaGetLastError",[]);
#endif
    
    var _last_error = CU.cuda_errors.length > 0 ? CU.cuda_errors.pop() : 0;

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([_last_error],"","");
#endif

    return _last_error;

  },

  /**
   * Description
   * @method cudaGetErrorString (no direct equivalent)
   * @param {} err   - Enumerator error
   * @return string
   */
  cudaGetErrorString: function(err) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaGetErrorString",[err]);
#endif
    
    var _string = "";

    switch(err) {
      case 0  : _string = "cudaSuccess";                          break;  // 0
      case 1  : _string = "cudaErrorMissingConfiguration";        break;  // 1
      case 2  : _string = "cudaErrorMemoryAllocation";            break;  // 2
      case 3  : _string = "cudaErrorInitializationError";         break;  // 3
      case 4  : _string = "cudaErrorLaunchFailure";               break;  // 4
      case 5  : _string = "cudaErrorPriorLaunchFailure";          break;  // 5
      case 6  : _string = "cudaErrorLaunchTimeout";               break;  // 6
      case 7  : _string = "cudaErrorLaunchOutOfResources";        break;  // 7
      case 8  : _string = "cudaErrorInvalidDeviceFunction";       break;  // 8
      case 9  : _string = "cudaErrorInvalidConfiguration";        break;  // 9
      case 10 : _string = "cudaErrorInvalidDevice";               break; // 10
      case 11 : _string = "cudaErrorInvalidValue";                break; // 11
      case 12 : _string = "cudaErrorInvalidPitchValue";           break; // 12
      case 13 : _string = "cudaErrorInvalidSymbol";               break; // 13
      case 14 : _string = "cudaErrorMapBufferObjectFailed";       break; // 14
      case 15 : _string = "cudaErrorUnmapBufferObjectFailed";     break; // 15
      case 16 : _string = "cudaErrorInvalidHostPointer";          break; // 16
      case 17 : _string = "cudaErrorInvalidDevicePointer";        break; // 17
      case 18 : _string = "cudaErrorInvalidTexture";              break; // 18
      case 19 : _string = "cudaErrorInvalidTextureBinding";       break; // 19
      case 20 : _string = "cudaErrorInvalidChannelDescriptor";    break; // 20
      case 21 : _string = "cudaErrorInvalidMemcpyDirection";      break; // 21
      case 22 : _string = "cudaErrorAddressOfConstant";           break; // 22
      case 23 : _string = "cudaErrorTextureFetchFailed";          break; // 23
      case 24 : _string = "cudaErrorTextureNotBound";             break; // 24
      case 25 : _string = "cudaErrorSynchronizationError";        break; // 25
      case 26 : _string = "cudaErrorInvalidFilterSetting";        break; // 26
      case 27 : _string = "cudaErrorInvalidNormSetting";          break; // 27
      case 28 : _string = "cudaErrorMixedDeviceExecution";        break; // 28
      case 29 : _string = "cudaErrorCudartUnloading";             break; // 29
      case 30 : _string = "cudaErrorUnknown";                     break; // 30
      case 31 : _string = "cudaErrorNotYetImplemented";           break; // 31
      case 32 : _string = "cudaErrorMemoryValueTooLarge";         break; // 32
      case 33 : _string = "cudaErrorInvalidResourceHandle";       break; // 33
      case 34 : _string = "cudaErrorNotReady";                    break; // 34
      case 35 : _string = "cudaErrorInsufficientDriver";          break; // 35
      case 36 : _string = "cudaErrorSetOnActiveProcess";          break; // 36
      case 37 : _string = "cudaErrorInvalidSurface";              break; // 37
      case 38 : _string = "cudaErrorNoDevice";                    break; // 38
      case 39 : _string = "cudaErrorECCUncorrectable";            break; // 39
      case 40 : _string = "cudaErrorSharedObjectSymbolNotFound";  break; // 40
      case 41 : _string = "cudaErrorSharedObjectInitFailed";      break; // 41
      case 42 : _string = "cudaErrorUnsupportedLimit";            break; // 42
      case 43 : _string = "cudaErrorDuplicateVariableName";       break; // 43
      case 44 : _string = "cudaErrorDuplicateTextureName";        break; // 44
      case 45 : _string = "cudaErrorDuplicateSurfaceName";        break; // 45
      case 46 : _string = "cudaErrorDevicesUnavailable";          break; // 46
      case 47 : _string = "cudaErrorInvalidKernelImage";          break; // 47
      case 48 : _string = "cudaErrorNoKernelImageForDevice";      break; // 48
      case 49 : _string = "cudaErrorIncompatibleDriverContext";   break; // 49
      case 50 : _string = "cudaErrorPeerAccessAlreadyEnabled";    break; // 50
      case 51 : _string = "cudaErrorPeerAccessNotEnabled";        break; // 51
      case 52 : _string = "cudaErrorDeviceAlreadyInUse";          break; // 52
      case 53 : _string = "cudaErrorProfilerDisabled";            break; // 53
      case 54 : _string = "cudaErrorProfilerNotInitialized";      break; // 54
      case 55 : _string = "cudaErrorProfilerAlreadyStarted";      break; // 55
      case 56 : _string = "cudaErrorProfilerAlreadyStopped";      break; // 56
      case 57 : _string = "cudaErrorAssert";                      break; // 57
      case 58 : _string = "cudaErrorTooManyPeers";                break; // 58
      case 59 : _string = "cudaErrorHostMemoryAlreadyRegistered"; break; // 59
      case 60 : _string = "cudaErrorHostMemoryNotRegistered";     break; // 60
      case 61 : _string = "cudaErrorOperatingSystem";             break; // 61
      case 62 : _string = "cudaErrorStartupFailure";              break; // 62
      case 63 : _string = "cudaErrorApiFailureBase";              break; // 63
      default: _string = "Unknown error";
    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([_string],"","");
#endif
    
    var _buffer = _malloc(_string.length);

    writeAsciiToMemory(_string, _buffer);

    return _buffer;
  },

  /**
   * Description
   * @method cudaKernelCall (kernelname<<<blocksPerGrid, threadsPerBlock>>>(...))
   * @param {} kernel_name      - Kernel name
   * @param {} kernel_source    - Kernel Source
   * @param {} blocksPerGrid    - Blocks per grid
   * @param {} threadsPerBlock  - Threads per grid
   * @param {} ... variable parameter
   * @return cudaSuccess(0)
   */
  cudaRunKernel: function(kernel_name, kernel_source, options, work_dim, global_work_size, local_work_size, num_args, param) {
#if CU_GRAB_TRACE
    CU.cudaBeginStackTrace("cudaKernelCall",[kernel_name, kernel_source,  options, work_dim, global_work_size, local_work_size, num_args, param]);
#endif
    var _kernel_options = options != 0 ? Pointer_stringify(options) : ""; 

    var _kernel_name = Pointer_stringify(kernel_name); 

    var _kernel_source = Pointer_stringify(kernel_source); 

    var _kernel_converted = CU.convertCudaKernelToOpenCL(_kernel_source,_kernel_name);

    if (_kernel_options) {
      // Add space after -D
      _kernel_options = _kernel_options.replace(/-D/g, "-D ");

      // Remove all the multispace
      _kernel_options = _kernel_options.replace(/\s{2,}/g, " ");
    }

    try {

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(CU.cuda_objects[CU.cuda_context]+".createProgram",[_kernel_converted]);
#endif       

      var _program = CU.cuda_objects[CU.cuda_context].createProgram(_kernel_converted);

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(_program+".build",[CU.cuda_objects[CU.cuda_device],_kernel_options,null]);
#endif        
      
      _program.build([CU.cuda_objects[CU.cuda_device]],_kernel_options,null);

#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(_program+".createKernel",[_kernel_name]);
#endif        
      
      var _kernel = _program.createKernel(_kernel_name);

      for (var i = 0; i < num_args; i++) {
         
        var webCLKernelArgInfo
        if (navigator.userAgent.toLowerCase().indexOf('firefox') == -1) 
          webCLKernelArgInfo = _kernel.getArgInfo(i);
        else
          webCLKernelArgInfo = {'addressQualifier':'','typeName':'float'};

        if (webCLKernelArgInfo.addressQualifier == "local") {
          console.error("cudaRunKernel (local paramater) not yet implemented ...\n");
        } else {

          if (param[i] in CU.cuda_objects) {
            // WEBCL OBJECT ARG
#if CU_GRAB_TRACE
            CU.cudaCallStackTrace(_kernel+".setArg",[i,CU.cuda_objects[param[i]]]);
#endif   
            _kernel.setArg(i,CU.cuda_objects[param[i]]);

          } else {

            if (webCLKernelArgInfo.typeName == "int") {
#if CU_GRAB_TRACE
              CU.cudaCallStackTrace(_kernel+".setArg",[i,param[i]]);
#endif   
              _kernel.setArg(i,new Int32Array([param[i]]));

            } else if (webCLKernelArgInfo.typeName == "float") {
#if CU_GRAB_TRACE
              CU.cudaCallStackTrace(_kernel+".setArg",[i,param[i]]);
#endif   
              _kernel.setArg(i,new Float32Array([param[i]]));

            } else {
              console.error("cudaRunKernel ("+webCLKernelArgInfo.typeName+" paramater) not yet implemented ...\n");
            }
          }
        }
      }

      var global_work_offset = [];
      for (var i = 0; i < work_dim; i++) {
        global_work_offset.push(0);
      }


#if CU_GRAB_TRACE
      CU.cudaCallStackTrace(""+CU.cuda_objects[CU.cuda_command_queue]+".enqueueNDRangeKernel",[_kernel,work_dim,global_work_offset,global_work_size,local_work_size,[],CU.cuda_event]);
#endif          
      
      var _event = null;
      if (CU.cuda_profile_event) {
        _event = new WebCLEvent();
        CU.cuda_events.push(_event);
      }
      
      CU.cuda_objects[CU.cuda_command_queue].enqueueNDRangeKernel(_kernel,work_dim,global_work_offset,global_work_size,local_work_size,[],_event);  

      _program.release();

      _kernel.release();

    } catch (e) {

      if (_program) {
        var _buildError = _program.getBuildInfo(CU.cuda_objects[CU.cuda_device],cuda.PROGRAM_BUILD_LOG);
        console.error(_buildError);
      }

      var _error = CU.catchError(e);

      CU.cuda_errors.push(30 /* cudaErrorUnknown */);

#if CU_GRAB_TRACE
      CU.cudaEndStackTrace([0],"",e.message);
#endif

      return 0; /* cudaSuccess */

    }

#if CU_GRAB_TRACE
    CU.cudaEndStackTrace([0],"","");
#endif

    return 0; /* cudaSuccess */
  },
  /**
   * Description
   * @method cudaKernelCallNoDim (kernelname<<<blocksPerGrid, threadsPerBlock>>>(...))
   * @param {} kernel_name      - Kernel name
   * @param {} kernel_source    - Kernel Source
   * @param {} blocksPerGrid    - Blocks per grid (int)
   * @param {} threadsPerBlock  - Threads per grid (int)
   * @param {} ... strut parameter
   * @return cudaSuccess(0)
   */
  cudaRunKernelFunc__deps: ['cudaRunKernel'],
  cudaRunKernelFunc: function(kernel_name, kernel_source, options, blocksPerGrid, threadsPerBlock, params) {
      
    var _local_work_size = [];
    var _global_work_size = [];
    var _param = [];

    _local_work_size[0] = threadsPerBlock;
    _global_work_size[0] = _local_work_size[0] * blocksPerGrid;

    var num_args = {{{ makeGetValue('params', '0', 'i32') }}};
    
    for (var i = 1 + num_args; i < 2 * num_args + 1; i ++) {
      var _typeoffset = i - num_args;
      var _type = CU.cuda_types[{{{ makeGetValue('params', '_typeoffset*4', 'i32') }}}];

      switch(_type){
        case 'i32':
          _param.push({{{ makeGetValue('params', 'i*4', 'i32') }}});
          
          break;
        case 'float':
          _param.push({{{ makeGetValue('params', 'i*4', 'float') }}});
          
          break;
        default:
          console.error("cudaRunKernelFunc : unknow type"); 
      }
    }

    _cudaRunKernel(kernel_name, kernel_source, options, 1, _global_work_size, _local_work_size, num_args, _param);

  },
  /**
   * Description
   * @method cudaKernelCallNoDim (kernelname<<<blocksPerGrid, threadsPerBlock>>>(...))
   * @param {} kernel_name      - Kernel name
   * @param {} kernel_source    - Kernel Source
   * @param {} blocksPerGrid    - Blocks per grid (dim)
   * @param {} threadsPerBlock  - Threads per grid (dim)
   * @param {} ... strut parameter
   * @return cudaSuccess(0)
   */
  cudaRunKernelDimFunc__deps: ['cudaRunKernel'],
  cudaRunKernelDimFunc: function(kernel_name, kernel_source, options, blocksPerGrid, threadsPerBlock, params) {
      
    var _local_work_size = [];
    var _global_work_size = [];
    var _param = [];

    for (var i = 0; i < 3; i ++) {
      _threadsPerBlock = {{{ makeGetValue('threadsPerBlock', 'i*4', 'i32') }}};
      _blocksPerGrid = {{{ makeGetValue('blocksPerGrid', 'i*4', 'i32') }}};

      _local_work_size[i] = _threadsPerBlock;
      _global_work_size[i] = _local_work_size[i] * _blocksPerGrid;
    }

    var num_args = {{{ makeGetValue('params', '0', 'i32') }}};
    
    for (var i = 1 + num_args; i < 2 * num_args + 1; i ++) {
      var _typeoffset = i - num_args;
      var _type = CU.cuda_types[{{{ makeGetValue('params', '_typeoffset*4', 'i32') }}}];

      switch(_type){
        case 'i32':
          _param.push({{{ makeGetValue('params', 'i*4', 'i32') }}});
          
          break;
        case 'float':
          _param.push({{{ makeGetValue('params', 'i*4', 'float') }}});
          
          break;
        default:
          console.error("cudaRunKernelFunc : unknow type"); 
      }
    }

    _cudaRunKernel(kernel_name, kernel_source, options, 3, _global_work_size, _local_work_size, num_args, _param);

  },
};

autoAddDeps(LibraryCUDA, '$CU');
mergeInto(LibraryManager.library, LibraryCUDA);



