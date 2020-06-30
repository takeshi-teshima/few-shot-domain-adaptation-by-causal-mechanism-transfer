Search.setIndex({docnames:["causal_da","causal_da.algorithm","causal_da.api_support","causal_da.api_support.logging","causal_da.api_support.validator","causal_da.components","causal_da.components.aug_predictor","causal_da.components.ica_torch","causal_da.components.inn_torch","causal_da.components.inn_torch.layers","causal_da.components.inn_torch.wrappers","causal_da.contrib","causal_da.contrib.evaluators_support","causal_da.contrib.evaluators_support.metrics","index","modules","setup"],envversion:55,filenames:["causal_da.rst","causal_da.algorithm.rst","causal_da.api_support.rst","causal_da.api_support.logging.rst","causal_da.api_support.validator.rst","causal_da.components.rst","causal_da.components.aug_predictor.rst","causal_da.components.ica_torch.rst","causal_da.components.inn_torch.rst","causal_da.components.inn_torch.layers.rst","causal_da.components.inn_torch.wrappers.rst","causal_da.contrib.rst","causal_da.contrib.evaluators_support.rst","causal_da.contrib.evaluators_support.metrics.rst","index.rst","modules.rst","setup.rst"],objects:{"":{causal_da:[0,0,0,"-"]},"causal_da.ICML2020_api":{CausalMechanismTransferICML2020API:[0,1,1,""]},"causal_da.ICML2020_api.CausalMechanismTransferICML2020API":{run_method_and_eval:[0,2,1,""]},"causal_da.algorithm":{api:[1,0,0,"-"],ica_augmenter:[1,0,0,"-"]},"causal_da.algorithm.api":{CausalMechanismTransfer:[1,1,1,""]},"causal_da.algorithm.api.CausalMechanismTransfer":{train:[1,2,1,""],train_and_record:[1,2,1,""]},"causal_da.algorithm.ica_augmenter":{ICAAugmenter:[1,1,1,""],ICATransferAugmenter:[1,1,1,""],full_combination:[1,3,1,""],get_size:[1,3,1,""],randint:[1,3,1,""],stochastic_combination:[1,3,1,""]},"causal_da.algorithm.ica_augmenter.ICAAugmenter":{augment:[1,2,1,""],augment_to_size:[1,2,1,""],is_valid_generated_data:[1,2,1,""]},"causal_da.algorithm.ica_augmenter.ICATransferAugmenter":{augment_to_size:[1,2,1,""],fit:[1,2,1,""]},"causal_da.api_support":{evaluator:[2,0,0,"-"],evaluator_runner:[2,0,0,"-"],index_class_label:[2,0,0,"-"],logging:[3,0,0,"-"],validator:[4,0,0,"-"]},"causal_da.api_support.evaluator":{AugmenterEvaluatorBase:[2,1,1,""],AugmenterValidationScoresEvaluator:[2,1,1,""],EvaluatorBase:[2,1,1,""]},"causal_da.api_support.evaluator.AugmenterEvaluatorBase":{evaluate:[2,2,1,""],set_augmenter:[2,2,1,""]},"causal_da.api_support.evaluator.AugmenterValidationScoresEvaluator":{evaluate:[2,2,1,""]},"causal_da.api_support.evaluator.EvaluatorBase":{evaluate:[2,2,1,""],get_save_results_type:[2,2,1,""],save_results_dict:[2,2,1,""]},"causal_da.api_support.evaluator_runner":{DummyEvaluatorRunner:[2,1,1,""],EvaluatorRunner:[2,1,1,""]},"causal_da.api_support.evaluator_runner.DummyEvaluatorRunner":{is_epoch_to_run:[2,2,1,""]},"causal_da.api_support.evaluator_runner.EvaluatorRunner":{is_epoch_to_run:[2,2,1,""]},"causal_da.api_support.index_class_label":{index_class_label:[2,3,1,""]},"causal_da.api_support.logging":{base:[3,0,0,"-"],mlflow:[3,0,0,"-"],model_logger:[3,0,0,"-"],mongo_sacred:[3,0,0,"-"],param_history_manager:[3,0,0,"-"],sacred:[3,0,0,"-"]},"causal_da.api_support.logging.base":{DummyRunLogger:[3,1,1,""],JointRunLogger:[3,1,1,""],RunLogger:[3,1,1,""]},"causal_da.api_support.logging.base.DummyRunLogger":{end_run:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""]},"causal_da.api_support.logging.base.JointRunLogger":{end_run:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""]},"causal_da.api_support.logging.base.RunLogger":{end_run:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""]},"causal_da.api_support.logging.mlflow":{MLFlowRunLogger:[3,1,1,""]},"causal_da.api_support.logging.mlflow.MLFlowRunLogger":{end_run:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""]},"causal_da.api_support.logging.model_logger":{MLFlowModelLogger:[3,1,1,""],MLFlowModelSaver:[3,1,1,""],MLFlowMultiModelLogger:[3,1,1,""]},"causal_da.api_support.logging.model_logger.MLFlowModelLogger":{load:[3,2,1,""],save:[3,2,1,""]},"causal_da.api_support.logging.model_logger.MLFlowModelSaver":{load:[3,2,1,""],load_zip:[3,4,1,""],save:[3,2,1,""]},"causal_da.api_support.logging.model_logger.MLFlowMultiModelLogger":{save:[3,2,1,""]},"causal_da.api_support.logging.mongo_sacred":{MongoAndSacredRunLogger:[3,1,1,""],sanitize_data:[3,3,1,""]},"causal_da.api_support.logging.mongo_sacred.MongoAndSacredRunLogger":{end_run:[3,2,1,""],get_current_run_id:[3,2,1,""],get_tags:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],perform_run:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""],update_mongo:[3,2,1,""]},"causal_da.api_support.logging.param_history_manager":{PandasParamHistoryManager:[3,1,1,""],ParamHistoryManagerBase:[3,1,1,""]},"causal_da.api_support.logging.param_history_manager.PandasParamHistoryManager":{filter:[3,2,1,""]},"causal_da.api_support.logging.param_history_manager.ParamHistoryManagerBase":{filter:[3,2,1,""]},"causal_da.api_support.logging.sacred":{SacredRunLogger:[3,1,1,""]},"causal_da.api_support.logging.sacred.SacredRunLogger":{end_run:[3,2,1,""],log_artifact:[3,2,1,""],log_metrics:[3,2,1,""],log_params:[3,2,1,""],log_params_exp_wide:[3,2,1,""],perform_run:[3,2,1,""],set_tags:[3,2,1,""],set_tags_exp_wide:[3,2,1,""],start_run:[3,2,1,""]},"causal_da.api_support.validator":{base:[4,0,0,"-"],performance:[4,0,0,"-"],scores:[4,0,0,"-"],timer:[4,0,0,"-"],util:[4,0,0,"-"]},"causal_da.api_support.validator.base":{DummyValidationScorer:[4,1,1,""],LossBasedValidationScorerBase:[4,1,1,""],ValidationScorerBase:[4,1,1,""]},"causal_da.api_support.validator.base.DummyValidationScorer":{evaluate:[4,2,1,""]},"causal_da.api_support.validator.base.LossBasedValidationScorerBase":{evaluate:[4,2,1,""]},"causal_da.api_support.validator.base.ValidationScorerBase":{evaluate:[4,2,1,""]},"causal_da.api_support.validator.performance":{SingleTargetDomainCVPerformanceValidationScorer:[4,1,1,""]},"causal_da.api_support.validator.performance.SingleTargetDomainCVPerformanceValidationScorer":{evaluate:[4,2,1,""]},"causal_da.api_support.validator.scores":{AugScoreBase:[4,1,1,""],AugSklearnScore:[4,1,1,""]},"causal_da.api_support.validator.timer":{Timer:[4,1,1,""]},"causal_da.api_support.validator.timer.Timer":{set:[4,4,1,""]},"causal_da.api_support.validator.util":{SingleRandomSplit:[4,1,1,""],SplitBase:[4,1,1,""],TargetSizeKFold:[4,1,1,""]},"causal_da.api_support.validator.util.SplitBase":{split:[4,2,1,""]},"causal_da.components":{aug_predictor:[6,0,0,"-"],ica_torch:[7,0,0,"-"],inn_torch:[8,0,0,"-"]},"causal_da.components.aug_predictor":{aug_gpr:[6,0,0,"-"],aug_krr:[6,0,0,"-"],base:[6,0,0,"-"],krr_params:[6,0,0,"-"],partial_loo_krr:[6,0,0,"-"],util:[6,0,0,"-"]},"causal_da.components.aug_predictor.aug_gpr":{AugGPR:[6,3,1,""],MLAllAugGPR:[6,1,1,""]},"causal_da.components.aug_predictor.aug_gpr.MLAllAugGPR":{fit:[6,2,1,""],predict:[6,2,1,""]},"causal_da.components.aug_predictor.aug_krr":{AugKRR:[6,1,1,""]},"causal_da.components.aug_predictor.aug_krr.AugKRR":{fit:[6,2,1,""]},"causal_da.components.aug_predictor.base":{AugPredictorBase:[6,1,1,""]},"causal_da.components.aug_predictor.base.AugPredictorBase":{fit:[6,2,1,""],predict:[6,2,1,""]},"causal_da.components.aug_predictor.partial_loo_krr":{PartialLOOCVKRR:[6,1,1,""],median_heuristic:[6,3,1,""]},"causal_da.components.aug_predictor.partial_loo_krr.PartialLOOCVKRR":{fit:[6,2,1,""],get_heuristic_gamma:[6,2,1,""],get_selected_params:[6,2,1,""],predict:[6,2,1,""]},"causal_da.components.aug_predictor.util":{Timer:[6,1,1,""]},"causal_da.components.aug_predictor.util.Timer":{set:[6,4,1,""]},"causal_da.components.ica_torch":{GCL_nonlinear_ica_train:[7,0,0,"-"],api:[7,0,0,"-"],gcl_model:[7,0,0,"-"],logging_util:[7,0,0,"-"],trainer_util:[7,0,0,"-"]},"causal_da.components.ica_torch.GCL_nonlinear_ica_train":{GCLTrainer:[7,1,1,""],GCL_nonlinear_ica_train:[7,3,1,""]},"causal_da.components.ica_torch.GCL_nonlinear_ica_train.GCLTrainer":{compute_and_backward_loss:[7,2,1,""],log_training_loss:[7,2,1,""]},"causal_da.components.ica_torch.api":{GCLTrainableInvertibleICAModel:[7,1,1,""]},"causal_da.components.ica_torch.api.GCLTrainableInvertibleICAModel":{set_train_params:[7,2,1,""],train_and_record:[7,2,1,""]},"causal_da.components.ica_torch.gcl_model":{ComponentWiseTransformWithAuxSelection:[7,1,1,""],ComponentwiseTransform:[7,1,1,""],GeneralizedContrastiveICAModel:[7,1,1,""]},"causal_da.components.ica_torch.gcl_model.GeneralizedContrastiveICAModel":{classify:[7,2,1,""],extract:[7,2,1,""],forward:[7,2,1,""],hidden:[7,2,1,""],inv:[7,2,1,""]},"causal_da.components.ica_torch.logging_util":{DummyRunLogger:[7,1,1,""]},"causal_da.components.ica_torch.logging_util.DummyRunLogger":{end_run:[7,2,1,""],log_artifact:[7,2,1,""],log_metrics:[7,2,1,""],log_params:[7,2,1,""],log_params_exp_wide:[7,2,1,""],set_tags:[7,2,1,""],set_tags_exp_wide:[7,2,1,""],start_run:[7,2,1,""]},"causal_da.components.ica_torch.trainer_util":{Log1pLoss:[7,1,1,""],random_pick_wrong_target:[7,3,1,""]},"causal_da.components.inn_torch":{glow_nonexponential:[8,0,0,"-"],layers:[9,0,0,"-"],wrappers:[10,0,0,"-"]},"causal_da.components.inn_torch.glow_nonexponential":{GlowNonExponential:[8,1,1,""],NN:[8,1,1,""]},"causal_da.components.inn_torch.glow_nonexponential.GlowNonExponential":{forward:[8,2,1,""],inv:[8,2,1,""],randomize_weights:[8,2,1,""]},"causal_da.components.inn_torch.glow_nonexponential.NN":{forward:[8,2,1,""],randomize_weights:[8,2,1,""]},"causal_da.components.inn_torch.layers":{actnorm:[9,0,0,"-"],affine_coupling_layer:[9,0,0,"-"],invertible_PLU:[9,0,0,"-"],invertible_linear:[9,0,0,"-"],pytorch_solve:[9,3,1,""]},"causal_da.components.inn_torch.layers.actnorm":{ActNorm:[9,1,1,""],mean:[9,3,1,""]},"causal_da.components.inn_torch.layers.actnorm.ActNorm":{forward:[9,2,1,""],initialize_parameters:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.layers.affine_coupling_layer":{AffineCouplingLayer:[9,1,1,""],NonexponentialAffineCouplingLayer:[9,1,1,""]},"causal_da.components.inn_torch.layers.affine_coupling_layer.AffineCouplingLayer":{forward:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.layers.affine_coupling_layer.NonexponentialAffineCouplingLayer":{forward:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.layers.invertible_PLU":{Inver:[9,1,1,""],InvertiblePLU:[9,1,1,""]},"causal_da.components.inn_torch.layers.invertible_PLU.Inver":{forward:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.layers.invertible_PLU.InvertiblePLU":{forward:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.layers.invertible_linear":{InvertibleLinear:[9,1,1,""],pytorch_solve:[9,3,1,""]},"causal_da.components.inn_torch.layers.invertible_linear.InvertibleLinear":{forward:[9,2,1,""],inv:[9,2,1,""]},"causal_da.components.inn_torch.wrappers":{sequential_flow:[10,0,0,"-"]},"causal_da.components.inn_torch.wrappers.sequential_flow":{SequentialFlow:[10,1,1,""]},"causal_da.components.inn_torch.wrappers.sequential_flow.SequentialFlow":{forward:[10,2,1,""],inv:[10,2,1,""]},"causal_da.contrib":{evaluators:[11,0,0,"-"],evaluators_support:[12,0,0,"-"]},"causal_da.contrib.evaluators":{AugmentingMultiAssessmentEvaluator:[11,1,1,""],ModelSavingEvaluator:[11,1,1,""],TargetDomainsAverageEvaluator:[11,1,1,""]},"causal_da.contrib.evaluators.AugmentingMultiAssessmentEvaluator":{evaluate:[11,2,1,""]},"causal_da.contrib.evaluators.TargetDomainsAverageEvaluator":{evaluate:[11,2,1,""]},"causal_da.contrib.evaluators_support":{aug_emd_assessment:[12,0,0,"-"],aug_gpr_assessment:[12,0,0,"-"],aug_info_assessment:[12,0,0,"-"],aug_krr_assessment:[12,0,0,"-"],base:[12,0,0,"-"],metrics:[13,0,0,"-"],util:[12,0,0,"-"]},"causal_da.contrib.evaluators_support.aug_emd_assessment":{AugEMDAssessment:[12,1,1,""]},"causal_da.contrib.evaluators_support.aug_gpr_assessment":{AugGPRAssessment:[12,1,1,""]},"causal_da.contrib.evaluators_support.aug_info_assessment":{AugmenterInfoAssessment:[12,1,1,""]},"causal_da.contrib.evaluators_support.aug_krr_assessment":{AugKRRAssessment:[12,1,1,""]},"causal_da.contrib.evaluators_support.base":{AssessmentBase:[12,1,1,""],AugAssessmentBase:[12,1,1,""],ResultTransformer:[12,1,1,""],StandardAssessmentBase:[12,1,1,""]},"causal_da.contrib.evaluators_support.metrics":{emd:[13,0,0,"-"],fractional_absolute_error:[13,0,0,"-"],mean_absolute_relative_error:[13,0,0,"-"],mmd:[13,0,0,"-"],mmd_two_sample_test:[13,0,0,"-"]},"causal_da.contrib.evaluators_support.metrics.emd":{EMD:[13,1,1,""]},"causal_da.contrib.evaluators_support.metrics.fractional_absolute_error":{fractional_absolute_error:[13,3,1,""]},"causal_da.contrib.evaluators_support.metrics.mean_absolute_relative_error":{mean_absolute_relative_error:[13,3,1,""]},"causal_da.contrib.evaluators_support.metrics.mmd":{MMD2u:[13,3,1,""],MMD:[13,1,1,""]},"causal_da.contrib.evaluators_support.metrics.mmd_two_sample_test":{MMD2u:[13,3,1,""],MMDH1:[13,1,1,""]},"causal_da.contrib.evaluators_support.util":{ArtifactLoggable:[12,1,1,""]},"causal_da.contrib.evaluators_support.util.ArtifactLoggable":{batch_save_artifact:[12,2,1,""],save_artifact:[12,2,1,""]},causal_da:{ICML2020_api:[0,0,0,"-"],algorithm:[1,0,0,"-"],api_support:[2,0,0,"-"],components:[5,0,0,"-"],contrib:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:classmethod"},terms:{"00000e":9,"0x7f0188bd2bf8":[],"0x7f1024055bf8":[],"0x7f1d4f3c5ae8":[],"0x7f4446be8ae8":[],"0x7f57e07bbbf8":12,"0x7f843e182b70":[],"0x7f882e4fdbf8":[],"0x7fa15d46fb70":[],"0x7fae38dec6a8":[],"0x7fde08b83ae8":[],"0x7fe954eaebf8":[],"0x7ff24e69db70":[],"2_u":13,"case":1,"class":[0,1,2,3,4,6,7,8,9,10,11,12,13],"default":[1,7],"float":[1,7],"function":[1,3,4,6,7,8,9,10,12],"import":3,"int":[0,1,7,9],"new":1,"return":[1,9],"true":[1,4,7],"while":[6,7,8,9,10],The:[0,1,3,6,7,13],_loss_1:7,_path:[3,7],_split:9,abov:1,activ:[0,9],actnorm:[0,5,8],adopt:9,affine_coupling_lay:[0,5,8],affinecouplinglay:9,after:[7,9],afterward:[7,8,9,10],algorithm:[0,2,15],all:[0,3,7,8,9,10],along:3,alpha:[6,12],also:[0,1,7],although:[7,8,9,10],analyt:6,api:[0,5,15],api_support:[0,1,11,15],appear:3,appera:3,appropri:1,arg:[4,6],arr:2,arrai:[1,7],artifact_subdir:[3,12],artifact_temp_dir:3,artifactlogg:12,assess:[11,12],assessmentbas:12,assum:[0,3],aug_emd_assess:[0,11],aug_gpr:[0,5],aug_gpr_assess:[0,11],aug_info_assess:[0,11],aug_krr:[0,5],aug_krr_assess:[0,11],aug_max_it:1,aug_predictor:[0,4,5],augassessmentbas:12,augemdassess:12,auggpr:6,auggprassess:12,augkrr:6,augkrrassess:12,augment:[1,2,6,11],augment_s:[4,6,11,12],augment_to_s:1,augmentation_s:1,augmenter_evalu:[0,1],augmenterevalu:1,augmenterevaluatorbas:[2,11],augmenterinfoassess:12,augmentervalidationscoresevalu:2,augmentingmultiassessmentevalu:11,augpredictorbas:[4,6],augscorebas:4,augsklearnscor:4,auto:1,aux_i:6,aux_x:6,auxiliari:7,balanc:7,base:[0,1,2,5,7,8,9,10,11,13],batch:9,batch_save_artifact:12,batch_siz:7,befor:[2,7],begin:7,behavior:1,best_score_model_logg:7,between:1,bia:9,blob:[9,13],bool:2,bound:1,box:0,broadcast:1,brute:0,build:1,byteord:1,c_src:[1,7],c_src_train:0,cache_s:1,call:[2,7,8,9,10],callabl:12,can:[7,9],candid:[0,3],care:[7,8,9,10],causal:1,causal_da:[14,15],causalmechanismtransf:1,causalmechanismtransfericml2020api:0,center:9,cfg_method:0,chaiyujin:9,channel:9,chunk:0,classif:7,classifi:7,classifier_hidden_dim:7,classifier_n_lay:7,classmethod:[3,4,6],close:1,code:[1,13],coef0:1,coeffici:7,com:[9,13],combin:0,compar:0,compon:[0,4,15],componentwise_transform:7,componentwisetransform:7,componentwisetransformwithauxselect:7,compress_format:3,comput:[7,8,9,10,13],compute_and_backward_loss:7,configur:0,connect:0,consid:3,construct:0,contain:[1,3,7,9,10],content:[14,15],contigu:9,contrast:7,contrastive_coeff:7,contrastive_loss:7,contrib:[0,15],cpu:7,data:[1,6,7],data_numpy_arrai:7,data_run_id:3,data_tensor:7,databas:[0,3],datafram:3,db_kei:[3,11],decor:[4,6],default_rng:1,defin:[7,8,9,10],degre:1,depend:6,depth:8,design:0,desir:1,detector:1,develop:3,devic:[1,7],dic:[3,7],dict:[1,2,3,12],differ:[1,6],dim:[7,8,9],dimens:[1,7,9],dimension:7,discret:1,dist:9,distribut:1,doe:3,domain:1,drawn:1,dtype:1,dummyevaluatorrunn:2,dummyrunlogg:[3,7],dummyvalidationscor:4,each:7,ebar:1,emanuel:13,emd:[0,11,12],end_run:[3,7],epoch:[2,3,11],epoch_callback:7,equat:9,estim:1,evalu:[0,1,4,15],evaluator_runn:[0,15],evaluatorbas:2,evaluatorrunn:2,evaluators_support:[0,11],even:3,everi:[7,8,9,10],exampl:[1,3,7,9],except:0,exclus:1,exist:3,experi:0,experiment_nam:3,exponenti:0,extens:1,extract:7,factor:9,fals:[1,8,9,13],feature_extractor:1,file:12,filenam:12,filter:3,fit:[1,6],flow:10,flycheck_bas:[],flycheck_icml2020_api:[],flycheck_scor:[],folder_nam:[3,7],follow:0,forc:0,form:6,former:[7,8,9,10],forward:[7,8,9,10],fractional_absolute_error:[0,11,12],fractionalabsoluteerror:13,from:[1,9],from_dict:3,full:1,full_combin:1,fulli:1,func:[3,7],gamma:[1,6],gcl:[0,7],gcl_ica_model:7,gcl_model:[0,5],gcl_nonlinear_ica_train:[0,5],gcltrainableinvertibleicamodel:7,gcltrainer:7,gener:[1,6,7,10],generalizedcontrastiveicamodel:7,get_current_run_id:3,get_heuristic_gamma:6,get_save_results_typ:2,get_selected_param:6,get_siz:1,get_tag:3,github:[9,13],given:[1,3,9],global:3,glow:[0,9],glow_nonexponenti:[0,5],glownonexponenti:8,gpu:1,grid:0,half:1,have:[0,9],heurist:6,hidden:7,hidden_dim:7,high:1,highest:1,hook:[7,8,9,10],hp_select:6,html:13,http:[9,13],hyper:[0,6],hyperparamet:0,ica:[0,1,7],ica_augment:[0,2,15],ica_data:1,ica_intermediate_evalu:[0,1],ica_logg:1,ica_torch:[0,5],icaaugment:1,icatransferaugment:[1,2],icml2020_api:[14,15],ident:9,ignit:13,ignor:[7,8,9,10],implement:[1,7],improv:7,includ:6,include_origin:1,inclus:1,index:[7,14],index_class_label:[0,15],indic:7,inf:1,inform:[1,7],init:9,initi:9,initialize_paramet:9,inn:7,inn_torch:[0,5],input:[7,9,12],insid:[1,3],inside_train_data_support:1,instanc:[1,7,8,9,10],instead:[1,7,8,9,10],integ:1,integr:2,interfac:0,intermedi:[1,7],intermediate_evalu:7,interv:1,inv:[1,7,8,9,10],inver:9,invers:9,invert:[0,1,7],invertible_ica_model:1,invertible_linear:[0,5,8],invertible_plu:[0,5,8],invertiblelinear:9,invertibleplu:9,irrespect:9,is_epoch_to_run:2,is_valid_generated_data:1,iter:[1,3,13],its:3,jointrunlogg:3,just:7,keepdim:9,kei:[1,3],kernel:[0,1,6,13],kernel_funct:13,kernel_two_sample_test:13,krr_param:[0,5],kwarg:[1,4,6,13],label:7,label_numpy_arrai:7,labels_numpy_arrai:7,largest:1,later:0,latter:[7,8,9,10],layer:[0,5,7,8],layerslist:10,learn:[1,7],leav:3,like:[1,2,9],linear:[7,9],list:7,load:[3,6],load_zip:3,log1ploss:7,log:[0,2,9],log_artifact:[3,7],log_metr:[3,7],log_param:[3,7],log_params_exp_wid:[3,7],log_training_loss:7,logger:[3,7],logging_util:[0,5],logtyp:2,loocv:6,loss:7,lossbasedvalidationscorerbas:4,low:1,lower:1,lowest:1,main:[0,4,6],make:3,master:[9,13],matmul:9,matric:9,matrix:9,max_epoch:7,max_it:1,maximum:1,mean:9,mean_absolute_relative_error:[0,11,12],mean_squared_error:[4,12],mechan:1,median:6,median_heurist:6,method:[0,1,2,3],metric:[0,2,11,12],minibatch:9,mlallauggpr:6,mlflow:[0,2],mlflow_tracking_uri:3,mlflowmodellogg:3,mlflowmodelsav:3,mlflowmultimodellogg:3,mlflowrunlogg:3,mmd2u:13,mmd:[0,11,12],mmd_two_sample_test:[0,11,12],mmdh1:13,model:[0,1,3,4,7],model_logg:[0,2],modelsavingevalu:11,modul:[14,15],modulelist:7,mongo_sacr:[0,2],mongo_t:3,mongoandsacredrunlogg:[0,3],mongodb:0,more:9,mse:12,multipli:7,must:1,n_aux:7,n_dim:9,n_hidden:8,n_input:8,n_label:7,n_layer:7,n_out:8,n_sampl:[7,9],namedtupl:9,namespac:[2,11],nan:1,nativ:1,ndarrai:[1,7],need:[1,2,3,7,8,9,10],nest:3,net:9,network:[0,7,9],neural:[0,7,9],none:[0,1,2,3,6,7,9,11,13],nonexponentialaffinecouplinglay:9,nonlinear:[1,7],normal:[9,10],novelti:1,novelty_detector:1,number:[0,1,7],numer:7,numpi:[1,7],object:[0,1,2,3,4,6,7,12,13],observ:3,occur:3,omit:1,one:[1,7,8,9,10],oneclasssvm:1,onli:[1,3,6],open:1,optim:7,option:[0,1,7,9],order:9,org:13,orig_data:6,origin:[1,6,9],other:2,otherwis:1,out:[0,1,9],output:[1,9],overrid:2,overridden:[7,8,9,10],packag:[14,15],page:14,panda:3,pandasparamhistorymanag:3,parallel:0,parallel_split_index:0,param:[3,7,9],param_grid:3,param_history_manag:[0,2],paramet:[0,1,3,6,7,9],paramhistorymanagerbas:[0,3],params_dict:[3,7],paramt:0,partial_loo_krr:[0,5],partialloocvkrr:6,pass:[1,7,8,9,10],path:[3,11,12],pathlib:3,per:9,perform:[0,1,2,6,7,8,9,10],perform_run:3,pick:7,place:7,point:[1,7,12],potenti:2,predict:[1,6],predictor:[0,1,4,6],predictor_kwarg:12,predictor_model:1,previou:[0,3],probe:0,provid:1,pytorch:[1,7,9,13],pytorch_solv:9,quick:1,randint:1,randn:9,random:1,random_integ:1,random_pick_wrong_target:7,random_st:[1,13],randomize_weight:8,ratio:1,rbf:[0,1,13],recip:[7,8,9,10],record:[0,1,2,7],regist:[7,8,9,10],regress:[0,6,13],relev:2,remov:[2,3],repres:9,requir:1,respect:9,result:[0,1,2],resulttransform:12,return_hidden:7,ridg:[0,6],row:7,run:[0,3,7,8,9,10],run_id:0,run_interv:2,run_kwarg:0,run_logg:[0,2,3,7,11],run_method_and_ev:0,runlogg:3,sacr:[0,2],sacred_artifact:3,sacredrunlogg:3,sampl:[1,13],sample_weight:6,sanitize_data:3,save:[0,3,12],save_artifact:12,save_fn:12,save_model_path:[0,7],save_results_dict:2,scale:[8,9],score:[0,2],score_transform:12,search:[0,14],see:[0,1],select:[0,6],sequenti:10,sequential_flow:[0,5,8],sequentialflow:10,set:[0,3,4,6],set_augment:2,set_tag:[3,7],set_tags_exp_wid:[3,7],set_train_param:7,setter:3,setup:15,shape:[1,7],should:[1,7,8,9,10],shrink:1,shuffl:7,sign:1,signal:7,silent:[7,8,9,10],similar:1,sinc:[7,8,9,10],singl:1,singlerandomsplit:4,singletargetdomaincvperformancevalidationscor:4,size:[1,7,9],sklearn:6,solut:9,solv:9,solver:9,sourc:[1,7],specif:1,specifi:1,split:[0,4],splitbas:4,squar:9,src_data:[1,7],stabil:7,standardassessmentbas:12,start:1,start_run:[3,7],statist:13,step:[3,7],stochastic_combin:1,str:[2,3,11],stride:9,string:12,subclass:[7,8,9,10],submodul:[5,14,15],subpackag:[14,15],subset:3,superset:3,support:1,sure:3,system:9,tag:3,tags_dict:[3,7],take:[1,7,8,9,10,12],tar_t:11,tar_tr:11,tar_tr_i:4,tar_tr_x:4,target:7,target_train_s:4,targetdomainsaverageevalu:11,targetsizekfold:4,task:3,tensor:9,test:13,them:[7,8,9,10],thi:[1,2,3,7,8,9,10],those:0,through:0,timer:[0,2,6],todo:[2,3],tol:1,torch:[1,7,8,9,10],total:0,total_parallel_split:0,train:[0,1,6,7,9],train_and_record:[1,7],train_param:1,train_siz:4,trainabl:[1,7],trainer:7,trainer_util:[0,5],transfer:1,transpos:9,tupl:[1,7,9],two:13,type:[1,9],uint8:1,unbias:13,unexist:3,uniform:1,union:[1,3,7],unit:9,unless:1,update_mongo:3,upper:1,usag:2,use:[0,1],use_plu:8,used:[1,6],using:1,util:[0,2,5,11],val_data:4,val_loss:4,valid:[0,2],validation_scorers_dict:2,validationscorerbas:[2,4],valu:[1,3,7,13],varianc:9,verbos:[1,13],version:[1,6,7],via:0,visual_assess:[],weight_decai:7,whatev:2,where:[1,9],which:[0,1],whole:6,wip_api:[],wip_icml2020_api:[],with_acceptance_ratio:1,with_lat:1,within:[7,8,9,10],without:0,wrapper:[0,5,8],x_dim:7,x_src:1,x_src_train:0,x_te:11,x_tr:11,x_train:6,xbar:1,y_pred:13,y_src:1,y_src_train:0,y_te:11,y_tr:11,y_train:6,you:2,zero:9,zip_path:3},titles:["causal_da package","causal_da.algorithm package","causal_da.api_support package","causal_da.api_support.logging package","causal_da.api_support.validator package","causal_da.components package","causal_da.components.aug_predictor package","causal_da.components.ica_torch package","causal_da.components.inn_torch package","causal_da.components.inn_torch.layers package","causal_da.components.inn_torch.wrappers package","causal_da.contrib package","causal_da.contrib.evaluators_support package","causal_da.contrib.evaluators_support.metrics package","Welcome to Causal DA\u2019s documentation!","few-shot-domain-adaptation-by-causal-mechanism-transfer","setup module"],titleterms:{actnorm:9,adapt:15,affine_coupling_lay:9,algorithm:1,api:[1,7],api_support:[2,3,4],assess:[],aug_emd_assess:12,aug_gpr:6,aug_gpr_assess:12,aug_info_assess:12,aug_krr:6,aug_krr_assess:12,aug_predictor:6,base:[3,4,6,12],causal:[14,15],causal_da:[0,1,2,3,4,5,6,7,8,9,10,11,12,13],compon:[5,6,7,8,9,10],content:[0,1,2,3,4,5,6,7,8,9,10,11,12,13],contrib:[11,12,13],document:14,domain:15,emd:13,evalu:[2,11],evaluator_runn:2,evaluators_support:[12,13],few:15,flycheck_bas:[],flycheck_icml2020_api:[],flycheck_scor:[],fractional_absolute_error:13,gcl_model:7,gcl_nonlinear_ica_train:7,glow_nonexponenti:8,ica_augment:1,ica_torch:7,icml2020_api:0,index_class_label:2,indic:14,inn_torch:[8,9,10],invertible_linear:9,invertible_plu:9,krr_param:6,layer:9,log:3,logging_util:7,mean_absolute_relative_error:13,mechan:15,metric:13,mlflow:3,mmd:13,mmd_two_sample_test:13,model_logg:3,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,16],mongo_sacr:3,packag:[0,1,2,3,4,5,6,7,8,9,10,11,12,13],param_history_manag:3,partial_loo_krr:6,perform:4,predictor:[],sacr:3,score:4,sequential_flow:10,setup:16,shot:15,submodul:[0,1,2,3,4,6,7,8,9,10,11,12,13],subpackag:[0,2,5,8,11,12],tabl:14,timer:4,trainer_util:7,transfer:15,util:[4,6,12],valid:4,visual_assess:[],welcom:14,wip_api:[],wip_icml2020_api:[],wrapper:10}})