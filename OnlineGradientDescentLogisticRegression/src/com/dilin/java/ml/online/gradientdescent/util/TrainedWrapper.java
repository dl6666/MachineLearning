package com.dilin.java.ml.online.gradientdescent.util;

public class TrainedWrapper {

	public double[] w;
	public double loss;
	public double[] wAve;
	public TrainedWrapper(double[] w, double loss, double[] wAve){
		this.w = w;
		this.loss = loss;
		this.wAve = wAve;
	}
 }
