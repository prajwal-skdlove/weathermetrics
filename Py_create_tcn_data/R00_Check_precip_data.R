library(data.table)
library(woeBinning)
library(nnet)

wd01 <- "~/Py_S4/Py_S4_v02_JHH/NCEI_parquet_files/"
stid <- "74486094789"  #JFK
stid <- "72278023183"  #Phoenix
stid <- "72202012839"  #Miami
stid <- "72243012960"  #Houston

postfix <- "_train.csv"
fnp <- paste(wd01,stid,postfix,sep="")
dt <- fread(fnp)


#Create a boolean target
dvn0 <- "tgt_bin"
dvn1 <- "tgt_01"
dtsub <- dt[,1:29]
#dtsub[[eval(dvn0)]] <- dt[['0']]
dtsub[,eval(dvn1) := ifelse(get(dvn0) < .4,0,1)]

table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]])

var_create <- lapply(1:8, function(x){
  nn1 <- paste("pct_stid_rain",x,sep="")
  on1 <- paste("cnt_gt_zero",x,sep="")
  on2 <- paste("cnt_ge_zero",x,sep="")
  dtsub[,eval(nn1) := 
          ifelse(get(on2) > 0,
                get(on1)/
                get(on2),NA)]
  return(1)
})

#WoE analysis
var = c(paste("avg",c(1,2,3,4,5,6,7,8),sep=""),
        paste("pct_stid_rain",c(1,2,3,4,5,6,7,8),sep=""))

woe_bins <- woe.binning(dtsub,eval(dvn1),var
                  ,min.perc.total=0.05
                  ,min.perc.class=0.001
                  ,stop.limit=0.1
                  )
woe_bins

#Deploy bins
df.with.binned.vars.added <- woe.binning.deploy(dtsub, woe_bins,
                                                add.woe.or.dum.var='woe')

#Results binomial model
dep_var = dvn1
vc <- c(2:4,5:29,32:39)
ind_var <- colnames(df.with.binned.vars.added)[vc]
reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))

nmodel <- multinom(reg_formula, data = na.omit(df.with.binned.vars.added))
p2 <- nmodel$fitted.values

conf_mat <- as.matrix(table(na.omit(df.with.binned.vars.added)[[dvn0]],round(p2,1)))
conf_mat

predpct_ge_50 <- as.vector(unlist(lapply(1:8, function(x) sum(conf_mat[x,6:11])/sum(conf_mat[x,]))))
predpct_lt_50 <- as.vector(unlist(lapply(1:8, function(x) sum(conf_mat[x,1:5])/sum(conf_mat[x,]))))
mres <- cbind(conf_mat,predpct_ge_50,predpct_lt_50)
mres

mon_rain <- as.matrix(table(df.with.binned.vars.added[["month_of_year"]],df.with.binned.vars.added$tgt_bin))
predpct_ge_5 <- as.vector(unlist(lapply(1:12, function(x) sum(mon_rain[x,6:8])/sum(mon_rain[x,]))))
predpct_le_1 <- as.vector(unlist(lapply(1:12, function(x) sum(mon_rain[x,1:4])/sum(mon_rain[x,]))))
mon_res <- cbind(mon_rain,predpct_ge_5,predpct_le_1)
mon_res
