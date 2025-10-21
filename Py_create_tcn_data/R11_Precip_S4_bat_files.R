install.packages("arrow")
library(data.table)
library(arrow)
library(woeBinning)
library(nnet)
library(caret)


#Get list of current s$ files
fn_get_file_info <- function(){
  wd01 <- "~/Py_S4/NCEI_parquet_files/"
  fpattern <- "*DAYSUM_train.parquet"
  lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
  lst_data_files <- list.files(wd01,pattern=fpattern,full.names =TRUE)
  lst_file_size <- as.vector(Reduce(rbind,as.numeric(lapply(lst_data_files, function(xf){
    mbs <- file.size(xf)/(1024*1024)
    mbs
  }))))
  
  dt_infiles <- data.table('Fname'=lst_data_filename,
                           'Fpath'=lst_data_files,
                           'Fsize'=lst_file_size)
  dt_infiles[,stid := substr(Fname,1,11)]
  dt_infiles_sub <- dt_infiles[Fsize >= 100,]
  return(dt_infiles_sub)
  #lst_mod_files <- dt_infiles_sub$Fpath
  #lst_mod_stid <- substr(dt_infiles_sub$Fname,1,11)
}

#Get data table of data file info
dt_files <- fn_get_file_info()

output_file <- "my_data.parquet"
write_parquet(dt_files, sink = output_file)
jj <- read_parquet(dt_files, output_file)



#Run mnl for one station
fn_pqt_one_model <- function(fpn,woe_cutoff){
  
  dt <- data.table(read_parquet(fpn))
  #nlen <- 100
  dt01 <- dt  #[1:1000,1:nlen]
  dt01$rec_id <- 1:dim(dt01)[1]
  
  #colnames need to be updated as some old files did not have names correct
  ocolnames <- paste("X",colnames(dt01),sep="")
  ncolnames <- ocolnames
  
  strt <- match('Xtgt_bin',ncolnames) #month_of_year
  str0 <- match('X0',ncolnames) #month_of_year
  str1 <- match('X1',ncolnames) #hour_of_day
  strid <- match('Xrec_id',ncolnames) #record id
  
  if(!is.na(strt)){ncolnames[strt] <- 'tgt_bin' }
  if(!is.na(str0)){ncolnames[str0] <- 'month_of_year' }
  if(!is.na(str1)){ncolnames[str1] <- 'hour_of_day' }
  if(!is.na(strid)){ncolnames[strid] <- 'rec_id' }
  
  colnames(dt01) <- ncolnames
  ncols <- length(ncolnames)
  col_order <- c(1:3,ncols,4:(ncols-1))
  dt01 <- data.table(dt01[,..col_order])
  ncolnames <- colnames(dt01)
  nvar_names <- ncolnames[5:length(ncolnames)]
  
  dvn0 <- "tgt_bin"
  dvn1 <- "tgt_01"
  dtsub <- dt01
  
  #Create binary target for woe
  dtsub[,eval(dvn1) := ifelse(get(dvn0) <= .5,0,1)]
  
  print("Binary target vs target")
  print(table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]]))
  
  dtsub[,fct_hours := paste("VL",hour_of_day,sep="")]
  dtsub[,fct_months := paste("VL",month_of_year,sep="")]
  
  #Number of variables under consideration
  num_vars <- length(nvar_names)
  
  vc_vars <- num_vars #1:10
  lst_woe_results <- lapply(vc_vars, function(x) {
            curr_var <- nvar_names[x]
            woe_bins <- woe.binning(dtsub,
                            ,target.var=eval(dvn1)
                            ,pred.var=curr_var
                            ,min.perc.total=0.05
                            ,min.perc.class=0.001
                            ,stop.limit=0.1)
            if(woe_bins[[3]] >= woe_cutoff) {
                pred_vars <- c(colnames(dtsub)[4])
                keep_vars <- c(pred_vars,curr_var)
                dtsub2 <- dtsub[,..keep_vars]
            
                dtsub2.with.binned.vars.added <- 
                      woe.binning.deploy(dtsub[,..keep_vars], 
                      woe_bins,
                      add.woe.or.dum.var='woe')
                rtn_val <- dtsub2.with.binned.vars.added
            }else{
              rtn_val <- curr_var
              }
  })
  
  lst_woe_results_keep <- unlist((lapply(1:length(lst_woe_results), 
                      function(x) is.null(dim(lst_woe_results[[x]]))))==FALSE)
  
  lst_woe_results_w_values <- lst_woe_results[lst_woe_results_keep]
  
  pred_vars <- c(eval(dvn1),colnames(dtsub)[1:4])
  df_woe_binned <- Reduce(function(x,y) 
    merge(x,y,by="rec_id"),c(list(dtsub[,..pred_vars]),lst_woe_results_w_values))
  
# 
#   
#   print("WoE data frame formed")
#   
#   #Keep WoE scores above a specific threshold
#   keep <- woe_bins[,3] > woe_cutoff
#   woe_bins2 <- woe_bins[keep,]
# 
#   #Identify predictor variables
#   pred_vars <- colnames(dtsub)[1:4]
#   keep_vars <- c(pred_vars,unlist(woe_bins[,1]))
#   dtsub2 <- dtsub[,..keep_vars]
# 
#   #Check how many binned variables were kept
#   num_woe_vars <- 1:length(woe_bins2[,1])
#   
#   
#   lst_woe_binned_vars <- lapply(num_woe_vars, function(x) {
#       woe_var_name <- unlist(woe_bins2[x,][1])
#       keep <- c(pred_vars[4],woe_var_name)
#       dt02.with.binned.vars.added <- woe.binning.deploy(dtsub[,..keep], woe_bins2[x,],
#                   add.woe.or.dum.var='woe')
#       dt02.with.binned.vars.added
#   })
# 
# df_woe_binned <- Reduce(function(x,y) merge(x,y,by="rec_id"),c(list(dtsub[,c(1:4)]),lst_woe_binned_vars))

  if(TRUE==FALSE){  
  print("Model data set created")
  
  # #Results binomial model all records
  dep_var = dvn0
  new_var = colnames(df_woe_binned)
  ind_var <- new_var["woe" == substr(new_var,1,3)]
  #ind_var <- ind_var[1:50]
  
  reg_formula <- as.formula(paste(dep_var," ~ ", "month_of_year + ","hour_of_day + ",paste(ind_var, collapse="+")))
  
  dt <- na.omit(df_woe_binned)
  nmodel <- multinom(reg_formula, data = dt)
  
  pred_probs <- predict(nmodel,newdata=dt,type="probs")
  data_with_preds <- cbind(dt, pred_probs)
  
  rtn <- list(nmodel,data_with_preds)
  
  print("Model done")
}
  
  rtn_val <- df_woe_binned
  return(rtn_val)
}


Sys.time()
fpn <- dt_files[3,1]
woe_cutoff <- 0.4
res <- fn_pqt_one_model(fpn,woe_cutoff)
Sys.time()

dt <- read_parquet(fpn,as_data_frame=TRUE)



fn_mnl_csv <- function(fnp){
  #fnp <- lst_data_files[10]
  print(fnp)
  dt <- fread(fnp)

  #colnames need to be updated as some old files did not have names correct
  ocolnames <- colnames(dt)
  ncolnames <- ocolnames
  ocolnames[1:10]
  str1 <- match('0',ncolnames) #tgt_bin
  str2 <- match('6',ncolnames) #hour_of_day
  str3 <- match('7',ncolnames) #month_of_year
  
  if(!is.na(str1)){ncolnames[str1] <- 'tgt_bin' }
  if(!is.na(str2)){ncolnames[str2] <- 'hour_of_day' }
  if(!is.na(str3)){ncolnames[str3] <- 'month_of_year' }

  colnames(dt) <- ncolnames
  
  dvn0 <- "tgt_bin"
  dvn1 <- "tgt_01"
  dtsub <- dt[,1:29]
  
  #Create binary target for woe
  dtsub[,eval(dvn1) := ifelse(get(dvn0) <= .5,0,1)]
  
  print("Binary target vs target")
  print(table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]]))
  
  #Convert months, hours to factors
  dtsub[,fct_hours := as.factor(get('hour_of_day'))]
  dtsub[,fct_months := as.factor(get('month_of_year'))]
  
  #Check for periods where no rain has fallen anywhere in region
  #in the past 8 hours
  nper = 8
  nn1 <- paste("all_stid_amt_rain","",sep="")
  nn2 <- paste("bin_stid_have_rain","",sep="")
  var_name <- c(paste("avg",1:nper,sep=""))
  dtsub[, eval(nn1) := rowSums(.SD), .SDcols = var_name]
  dtsub[, eval(nn2) := ifelse(get(nn1)>0,1,0)]
  
  print("Periods where no rain fell (0) within 100km for preceding 8 hours")
  print(table(dtsub[[eval(nn2)]]))
  print(table(dtsub[[eval(nn2)]],dtsub[[eval(dvn0)]]))

  #Convert counts to pct of stations with rain
  nper = 8
  var_create <- lapply(1:nper, function(x){
    nn1 <- paste("pct_stid_rain",x,sep="")
    on1 <- paste("cnt_gt_zero",x,sep="")
    on2 <- paste("cnt_ge_zero",x,sep="")
    dtsub[,eval(nn1) := 
            ifelse(get(on2) > 0,
                   get(on1)/
                     get(on2),NA)]
    return(1)
  })
  
  dtsub[,diff_avg1_2 := avg1 - avg2]
  dtsub[,diff_avg2_3 := avg2 - avg3]
  dtsub[,diff_avg3_4 := avg3 - avg4]
  dtsub[,diff_avg4_5 := avg4 - avg5]
  dtsub[,diff_avg5_6 := avg5 - avg6]
  dtsub[,diff_avg6_7 := avg6 - avg7]
  dtsub[,diff_avg7_8 := avg7 - avg8]

  #WoE analysis
  lst_ivar <- c("hour_of_day",
                "month_of_year",
                nn1,
                nn2,
                eval(paste("pct_stid_rain",1:8,sep="")),
                eval(paste("avg",c(1,2,3,4,5,6,7,8),sep="")),
                eval(paste("diff_avg",c("1_2","2_3","3_4","4_5","6_7","7_8"),sep="")))
  
  woe_bins <- woe.binning(dtsub,eval(dvn1),lst_ivar
                          ,min.perc.total=0.05
                          ,min.perc.class=0.001
                          ,stop.limit=0.1
  )
  woe_bins
  
  #Deploy bins
  df.with.binned.vars.added <- woe.binning.deploy(dtsub, woe_bins,
                                                  add.woe.or.dum.var='woe')

  #Results binomial model all records
  dep_var = dvn1
  new_var = colnames(df.with.binned.vars.added)
  ind_var <- new_var["woe" == substr(new_var,1,3)]
  
  reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))
  
  dt <- na.omit(df.with.binned.vars.added)
  nmodel <- multinom(reg_formula, data = dt)
  
  pred_probs <- predict(nmodel,newdata=dt,type="probs")
  data_with_preds <- cbind(dt, pred_probs)
  
  rtn <- list(nmodel,data_with_preds)
  
  return(rtn)
}
fn_mnl_pqt <- function(fnp){
  #fnp <- lst_data_files[10]
  print(fnp)
  dt <- data.table(read_parquet(fnp))
  
  #colnames need to be updated as some old files did not have names correct
  ocolnames <- paste("X",colnames(dt),sep="")
  ncolnames <- ocolnames

  strt <- match('Xtgt_bin',ncolnames) #month_of_year
  str0 <- match('X0',ncolnames) #month_of_year
  str1 <- match('X1',ncolnames) #hour_of_day
  
  if(!is.na(strt)){ncolnames[strt] <- 'tgt_bin' }
  if(!is.na(str0)){ncolnames[str0] <- 'month_of_year' }
  if(!is.na(str1)){ncolnames[str1] <- 'hour_of_day' }
  
  colnames(dt) <- ncolnames
   
  dvn0 <- "tgt_bin"
  dvn1 <- "tgt_01"
  dtsub <- dt  #dt[,1:29]
   
  #Create binary target for woe
  dtsub[,eval(dvn1) := ifelse(get(dvn0) <= .5,0,1)]

  print("Binary target vs target")
  print(table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]]))



  #Identify rain measurements
  nmetrics = 6
  min_var_field <- 4
  max_var_field <- length(ocolnames)
  print(ncolnames[1:10])
  
  #Metric columns
  var_precip <- seq(min_var_field,max_var_field,nmetrics)
  var_temp <- seq(min_var_field+1,max_var_field,nmetrics)
  var_RH <- seq(min_var_field+2,max_var_field,nmetrics)
  var_RHmax <- seq(min_var_field+3,max_var_field,nmetrics)
  var_RHmin <- seq(min_var_field+4,max_var_field,nmetrics)
  
  lst_metric_cols <- c("var_precip",
                       "var_temp",
                       "var_RH",
                       "var_RHmax",
                       "var_RHmin")
  
  #lst_dt_metrics <- lapply(lst_metric_cols, function(x) dt[,get(x),with=FALSE])

  #WoE analysis
  #lst_ivar <- c(1,2,3,var_precip)
  vars <- colnames(dtsub)[4:1008]
  #Convert months, hours to factors
  dtsub[,fct_hours := paste("VL",hour_of_day,sep="")]
  dtsub[,fct_months := paste("VL",month_of_year,sep="")]
  
  #vars <- c("fct_months","fct_hours","X75","X81","X87")
  #dt_woe <-  data.frame(dtsub[,..vars])
  
  woe_bins <- woe.binning(dtsub,
                          ,target.var='tgt_01'
                          ,pred.var=vars
                          ,min.perc.total=0.05
                          ,min.perc.class=0.001
                          ,stop.limit=0.1
                          )
  
  
  woe_bins_tmp <- woe_bins[1:5,]
  
  # #Deploy bins
  df.with.binned.vars.added <- woe.binning.deploy(dtsub, woe_bins_tmp,
                                                   add.woe.or.dum.var='woe')
  
  #start <- 801
  #vc_int <- start:1005
  #woe_bins_tmp <- woe_bins[vc_int,]
  #woe_bins_tmp
  #df.with.binned.vars.added <- woe.binning.deploy(df.with.binned.vars.added, woe_bins_tmp,
  #                                                add.woe.or.dum.var='woe')
  
  
  
  # #Results binomial model all records
  dep_var = dvn1
  new_var = colnames(df.with.binned.vars.added)
  ind_var <- new_var["woe" == substr(new_var,1,3)]
  ind_var <- ind_var[1:50]
  
  reg_formula <- as.formula(paste(dep_var," ~ ", "month_of_year + ","hour_of_day + ",paste(ind_var, collapse="+")))
  
  dt <- na.omit(df.with.binned.vars.added)
  nmodel <- multinom(reg_formula, data = dt)
   
  pred_probs <- predict(nmodel,newdata=dt,type="probs")
  data_with_preds <- cbind(dt, pred_probs)
   
  rtn <- list(nmodel,data_with_preds)
  
  #rtn <- dtsub
  return(rtn)
}
#Get model output for all stations
fnp <- lst_mod_files[1]
lst_mnl_results <- lapply(lst_mod_files[1], function(xfn) fn_mnl_pqt(xfn))

jj <- data.table(data_with_preds)
jj[,prob_bkt := cut(pred_probs,c(0,.4,.5,.6,.7,.8,.9,1))]

table(jj[,2])
jjt <- jj[,'pred_probs']
jjbk <- c(0,.4,.5,.6,.7,.8,.9,1)
table(cut(as.numeric(unlist(jjt)),breaks=jjbk))
table(jj[,'tgt_bin'],jj[,'pred_probs'])
table(jj[,c("tgt_bin","prob_bkt")])

fn_confmat <- function(dtres,act_thresh,pred_thresh){
  
  dtres <- dtres[[2]]
  actual_values <- factor(dtres$tgt_bin >= act_thresh)
  predicted_values <- factor(dtres$pred_probs >= pred_thresh)
  cm <- confusionMatrix(data = predicted_values, 
                        reference = actual_values, 
                        positive = "TRUE")
  Precision <- cm$byClass[match("Precision",attr(cm$byClass,"names"))]
  Recall <- cm$byClass[match("Sensitivity",attr(cm$byClass,"names"))]
  attr(Recall,"names") <- "Recall"
  
  lst_rtn <- list(list(Precision,Recall),cm)
  return(lst_rtn)
  
}

dtres <- data_with_preds
vc_act <- c(.5,1,4)
vc_prb <- c(.1,.2,.3,.4,.5,.6,.7,.8,.9)
jj <- fn_confmat(dtres[,c('tgt_bin','pred_probs')],vc_act,vc_prb)

pp <- as.factor(dtres$pred_probs > .5)
ac <- as.factor(dtres$tgt_bin >= 0.5)

jjcm <- confusionMatrix(data = pp, 
                reference = ac, 
                positive = "TRUE")
jjcm

fn_confmat_grid <- function(fname,dtres,vc_act,vc_pred){
  # fname <- lst_mod_files[1]
  # dtres <- lst_mnl_results[[1]]
  # vc_act <- vc_act
  # vc_pred <- vc_prb
  
  lst_rtn <- lapply(vc_act, function(xact) {
    Reduce(rbind,lapply(vc_pred, function(xprd) {
      res <- fn_confmat(dtres,xact,xprd)
      rtn <- data.table(t(res[[1]]))
  
      colnames(rtn) <- c("Precision","Recall")

      rtn$Actual_Threshold <- xact
      rtn$Prob_Threshold <- xprd  
      rtn$Source <- fname
      rtn
    }))
     
    })

  dt_rtn <- Reduce(rbind,lst_rtn)
  
  return(dt_rtn)
}

vc_act <- c(.5,1,4)
vc_prb <- c(.1,.15,.18,.2,.22,.25,.3)

ilim <- length(lst_mod_files)
lst_grids <- lapply(1:ilim, function(xidx) {
  print(xidx)
  fn_confmat_grid(lst_mod_files[xidx],lst_mnl_results[[xidx]],vc_act,vc_prb)
})


dt_grid <- Reduce(rbind,lst_grids)
dt_grid[,fct_Prob_Threshold := factor(Prob_Threshold)]

dt_grid[,valueP := as.numeric(Precision)]
bx_Prec_med <- boxplot(valueP ~ fct_Prob_Threshold,data=dt_grid)
vc_Prec_med <- bx_Prec_med$stats[3,]

dt_grid[,valueR := as.numeric(Recall)]
bx_Recall_med <- boxplot(valueR ~ fct_Prob_Threshold,data=dt_grid)
vc_Recall_med <- bx_Recall_med$stats[3,]

xx <- vc_Prec_med*vc_Prec_med
yy <- vc_Recall_med*vc_Recall_med
zz <- sqrt(xx+yy)
zz

stids <- unique(dt_grid$Source)
stids


#Create bat file to run s4 models on same data files
lst_cmds <- unlist(lapply(3:length(stids), function(idx){
  #idx <- 1
  lst_mod_stid[idx]
  cmd_data_file <- stids[idx]
  iepochs <- 100
  bat_content <- paste("python -m s4model --modelname",
                     paste("s4",lst_mod_stid[idx],sep="_"),
                     "--modeltype classification --dataset ",
                     paste(substr(stids[idx],1,nchar(stids[idx])-4),".parquet",sep=""),
                     " --tabulardata --dependent_variable tgt_bin --epochs ",
                     iepochs,
                     sep=" ")
  bat_content
  }))
file_conn <- file("C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\s4model\\py_s4b.bat",open="w")
writeLines(lst_cmds,con=file_conn)
close(file_conn)


#Get list of current s4 output files
wd01 <- "C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\results"
fpattern <- "*Test_results*"
lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
lst_data_fullname <- list.files(wd01,pattern=fpattern,full.names =TRUE)

ix <- 20

lst_s4_cm <- lapply(1:length(lst_data_filename),function(ix){
  dt_res <- fread(file=lst_data_fullname[ix])
  prd <- as.factor(dt_res$Predicted)
  act <- as.factor(dt_res$tgt_bin)
  
  cm <- confusionMatrix(data = prd, 
                        reference = act,
                        mode="everything")
  
  cm[[6]] <- lst_data_filename[ix]
  cm[[7]] <- substr(cm[[6]],4,14)
  
  # rtn_summ <- cm[[4]][,c("Precision","Recall","Balanced Accuracy")]
  # rnames <-  unlist(attr(lst_s4_cm[[29]],"dimnames")[[1]])
  # st_id <- as.character(cm[[7]])
  
  rtn_sub <- cm[[4]][,c("Precision","Recall","Balanced Accuracy")]
  rtn_class <- unlist(attr(rtn_sub,"dimnames")[[1]])
  
  rtn_val <- data.table(rtn_sub,Class = rtn_class, St_id = as.character(cm[[7]]))
  rtn_val 
})
dt <- data.table(Reduce(rbind,lst_s4_cm))

