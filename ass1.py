#------------ASSIGMENT1-------------
"""
this is python implemation of collegue system

"""
Uniname="cairo uni"
print(f"My uni is {Uniname}")
#------------------------------------
#-------------LISTS------------------
uni_name=["Cairo_uni","AUC","GUC","Mansoura"]

for i in range(4):
    print(uni_name)
    
#---------------DIC----------------
course={"name":"ALGO","Duration":50,"Prereq":"DATA_STRUCTURE"}
print(course)
#_____________________________________
class UNI:
    def __init__(self,name,faculty,rank):
        self.name=name
        self.faculty=faculty
        self.rank=rank
        
    def print_uni_info(self):
        print(f"This is {self.name} university faculty of {self.faculty} has rank :{self.rank}")
    
    def count_rank(self):
        if self.rank >90:
            print("THis is cairo uni") 
        elif self.rank<90 & self.rank>70:
            print("This is GUC university")
        elif self.rank<70 & self.rank>50:
            print("This is AUC university")
        else :print("there is no uni for this low rank")
   
        
   
University=UNI("Cairo uni","Faculty of engineering",60)
University.count_rank()
University.print_uni_info()  
#--------------------------------------------------
i=0
while i<5:
    print("this is my wjile loop")
    i+=1
