#include<cstring>
#include<cstdlib>
#include<cstdio>
#include<cstdint>
#include<elf.h>
#include<cassert>

void readelf(char *filename){
    FILE* fd = fopen(filename,"r");
    if(!fd){
        printf("can't find %s\n",filename);
    }
    Elf64_Ehdr ehdr;
    //Elf64_Phdr phdr;
    Elf64_Shdr shdr,strtab_shdr;
    char *strtab = NULL; 
    assert(fseek(fd,0,SEEK_SET)==0);
    assert(fread(&ehdr,1,sizeof(Elf64_Ehdr),fd)==sizeof(Elf64_Ehdr));
    assert(*(uint32_t *)ehdr.e_ident==0xba55ed50);
    
    // search strtab
    for(int i=ehdr.e_shnum-1;i>=0;i--){
        assert(fseek(fd,ehdr.e_shoff+i*sizeof(Elf64_Shdr),SEEK_SET)==0);
        assert(fread(&shdr,1,sizeof(Elf64_Shdr),fd)==sizeof(Elf64_Shdr));
        printf("%d\n",shdr.sh_type);
        if(shdr.sh_type==SHT_STRTAB){
            strtab_shdr = shdr;
            assert(fseek(fd,strtab_shdr.sh_offset,SEEK_SET)==0);
            strtab = (char *)malloc(strtab_shdr.sh_size);
            assert(fread(strtab,1,strtab_shdr.sh_size,fd)==strtab_shdr.sh_size);
            break;
        }
    }
    for(int i=0;i<ehdr.e_shnum;i++){
        assert(fseek(fd,ehdr.e_shoff+i*sizeof(Elf64_Shdr),SEEK_SET)==0);
        assert(fread(&shdr,sizeof(Elf64_Shdr),1,fd)==1);
    }
}

int main(){
    char name[100] = "bin/dummy_fatbin";
    printf("reading %s\n",name);
    readelf(name);
}