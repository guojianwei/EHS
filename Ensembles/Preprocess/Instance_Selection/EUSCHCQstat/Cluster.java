package keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;

import java.util.ArrayList;
import java.util.List;

public class Cluster {
	int head;
	ArrayList<Integer> pointlist;
	int n;
	public Cluster(int head, ArrayList<Integer> pl, int n){
		this.head = head;
		this.pointlist = pl;
		this.n = n;
	}
	public static Cluster merger2Clst(Cluster x, Cluster y){
		int tempHead = x.getHead();
		if(x.length() > y.length()){
		    tempHead = y.getHead();
		}
		ArrayList<Integer> xlist = x.getPointList();
		ArrayList<Integer> ylist = y.getPointList();
		int tempN = x.length()+y.length();
		ArrayList<Integer>  tempPL = new ArrayList<Integer> ();
		tempPL.addAll(xlist);
		tempPL.addAll(ylist);
		return new Cluster(tempHead,tempPL,tempN);
	}
	ArrayList<Integer> getPointList()
	{
		return this.pointlist;
	}
    int getHead(){return this.head;}
    int length(){return this.n;}
        
}
