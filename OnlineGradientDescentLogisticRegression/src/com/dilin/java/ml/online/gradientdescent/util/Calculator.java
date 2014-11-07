package com.dilin.java.ml.online.gradientdescent.util;

public class Calculator {

	public static double getLoss(double innerProd, int y) {
		return Math.log(1.0 + Math.exp(innerProd)) - y*innerProd;
	}
	
	public static double[] getLambda(double[] x, double innerProd, int y) {
		double expid = 1.0 / (1.0 + Math.exp(-innerProd)) - y;
		double[] res = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			res[i] = x[i] * expid;
		}
		return res;
	}

	public static double getInnerProduct(double[] x, double[] w) {
		double innerProd = 0;
		for(int i = 0; i < x.length; i++){
			innerProd += x[i] * w[i];
		}
		return innerProd;
	}
}
