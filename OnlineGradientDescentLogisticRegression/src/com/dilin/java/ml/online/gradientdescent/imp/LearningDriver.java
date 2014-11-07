package com.dilin.java.ml.online.gradientdescent.imp;

import java.io.IOException;

import com.dilin.java.ml.online.gradientdescent.util.TrainedWrapper;

public class LearningDriver {

	public static void main(String[] args) throws NumberFormatException, IOException {
		String trainFile = "rcv1.train.vw";
		String testFile = "rcv1.test.vw";
		int trainLine = 65536;
		int testLine = 23149;
		int dim = 50000;
		double eta = 1.0;
		Learner l = new Learner();
		TrainedWrapper res = l.getTrained(trainFile, trainLine, dim, eta);
		System.out.println("Total Loss of learner divided by T is " + res.loss / trainLine);
		System.out.println("Train loss of w_t+1 is " + l.getLoss(res.w, trainFile, trainLine, dim, eta) / trainLine);
		System.out.println("Test loss of w_t+1 is " + l.getLoss(res.w, testFile, testLine, dim, eta) / testLine);
		System.out.println("Train loss of w_ave is " + l.getLoss(res.wAve, trainFile, trainLine, dim, eta) / trainLine);
		System.out.println("Test loss of w_ave is " + l.getLoss(res.wAve, testFile, testLine, dim, eta) / testLine);
		System.out.println("Error of w_T+1 is " + l.getClassificationError(res.w, testFile, testLine, dim, eta));
		System.out.println("Error of w_ave is " + l.getClassificationError(res.wAve, testFile, testLine, dim, eta));
	}
}
