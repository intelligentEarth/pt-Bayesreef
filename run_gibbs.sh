
#!/bin/sh 
echo Running all 	
 

  #minimum should be 2000 samples with swap of 0.01
 
for x in  1  
	do 	
			python ptBayesReef_Gibbs.py  1 10 1 50000
  
	done 



 