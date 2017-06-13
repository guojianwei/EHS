package keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;  
import java.util.Map;

import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.KNN;

import java.util.LinkedList;
import java.util.List;

/**
 *Hierarchical Clustering
 *
 */
public class HierarchicalClustering {
	private double[][] mergeClst(Map<Integer,Cluster> clsts,List<Integer> lastHead,double lastM[][],int uF[],double threshold){
	   	int n = lastM.length;
		double minAvgDis = 0;
		int row = 0, col = 0, i, j;
		for (i=0; i<n; i++)
			for(j=i+1; j<n; j++)
				if(lastM[i][j]>minAvgDis)
					minAvgDis = lastM[i][j];
	    
		for (i=0; i<n; i++)
			for(j=i+1; j<n; j++){
		        double avg = lastM[i][j]/(clsts.get(lastHead.get(i)).length()*clsts.get(lastHead.get(j)).length());
	            if (avg < minAvgDis){
	                minAvgDis = avg;
	                row = i; col = j;
	                if(col<row){int ti = row; row = col; col =ti;}
	            }
			}
	    if (minAvgDis > threshold){
	    	double res[][] = new double[1][1];
	    	res[0][0]=0;
	        return res;
	    }
	    int hx = lastHead.get(row), hy = lastHead.get(col);
	    double newM[][] = new double [n-1][n-1];
	    int oi,oj;
	    lastHead.remove(col);
	    lastHead.remove(row); // remove by indices 
	    for(i=0,oi=0; oi<n; oi++){
	    	if(oi!=row && oi!=col){
		    	for(oj=0,j=0; oj<n; oj++){
		    		if(oj!=row && oj!=col){
		    			newM[i][j] = lastM[oi][oj];
		    			j++;
		    		}
		    	}
		    	i++;
	    	}
	    }
	    for(i=0,oi=0; oi<n; oi++){
	    	if(oi!=row && oi!=col){
	    		double old2new = lastM[oi][row]+lastM[oi][col];
	    		newM[i][n-2] = newM[n-2][i] = old2new;
		     	i++;
	    	}
	    }
	    Cluster nclst = Cluster.merger2Clst(clsts.get(hx),clsts.get(hy));
	    
	    clsts.remove(hy);
	    clsts.remove(hx);
	    clsts.put(nclst.getHead(), nclst);
	    if (nclst.getHead() != hx){
	    	uF[hy] = uF[hx]+uF[hy];        
	    	uF[hx] = hy;
	    }else{
	    	uF[hx] = uF[hx]+uF[hy];
	    	uF[hy] = hx;
	    }
	    lastHead.add(nclst.getHead());
	    return newM;
	    }

	/**
	 * average-linkage agglomerative clustering, a hierarchical clustering process.
	 *@param points 2-dimensional sample list double
	 * @param resUF 
	 *@param cp Th = davg * Cp:
	 *@return The Euclidean distance
	 */
	public void HClustering(double points[][],int[] resUF, double cp){
		int n = points.length;
		double M[][] = new double[n][n];
		int i, j;
		for(i=0; i<n; i++)
		    for(j=0; j<n; j++)
		         M[i][j] = KNN.distancia(points[i],points[j]);
		Map<Integer, Cluster> clusters = new HashMap<Integer, Cluster>();   
		for(i=0; i<n; i++){
			ArrayList<Integer>tl = new ArrayList<Integer>();
			tl.add(i);
		    clusters.put(i,new Cluster(i,tl,1));
		}
		double avgMinDis = 0.0;
		for(i=0; i<n; i++){
		    double minDis = M[i][0]+M[i][n-1];
		    for(j=0; j<n; j++)
		        if(j!=i && M[i][j]<minDis)
		            minDis = M[i][j];
		    avgMinDis += minDis;
		}
		avgMinDis /= n;
		double threshold = avgMinDis*cp;
		List<Integer> headlist =new ArrayList<Integer>();
		int unionF[] = new int[n];
		for(i=0; i<n; i++){
			headlist.add(i);
			unionF[i] = -1;
		}
		while(M.length!=1)
		    M = mergeClst(clusters,headlist,M,unionF,threshold);
		for(i=0; i<n; i++){
			j = unionF[i];
			resUF[i] = i;
			while(j>=0)
			{
				resUF[i] = j;
				j = unionF[j];
			}
		}
	}  
	public double [][] readFileBylines(String fileName){
		ArrayList <ArrayList<Double> >List = new ArrayList<ArrayList<Double> >();
		File file = new File(fileName);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            int line = 0;
            while ((tempString = reader.readLine()) != null) {
                String slist[] = tempString.split(" ");
            	if(!slist[0].startsWith("@")){
                	ArrayList<Double> tempList = new ArrayList<Double>();
	                for(String x:slist)
	                	tempList.add(Double.valueOf(x));
	            	List.add(tempList);
            	}
                line++;
            }
            reader.close();
            double res[][] = new double[List.size()][List.get(0).size()];
            for(int i=0; i<List.size(); i++)
            {
            	ArrayList<Double> tempList = List.get(i);
            	for(int j=0; j<tempList.size(); j++)
            		res[i][j] = tempList.get(j);
            }
            return res;
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
        return new double[1][1];
	}

}  