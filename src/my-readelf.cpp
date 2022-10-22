#include<cstring>
#include<cstdlib>
#include<cstdio>
#include<cstdint>
#include<elf.h>
#include<cassert>

void readelf(char *filename,char *sec_name,uint8_t **nv_fatbin,unsigned *size){
    FILE* fd = fopen(filename,"r");
    if(!fd){
        printf("can't find %s\n",filename);
    }
    Elf64_Ehdr ehdr;
    //Elf64_Phdr phdr;
    Elf64_Shdr shdr,strtab_shdr;
    char *strtab = NULL; 
    *nv_fatbin = NULL;
    assert(fseek(fd,0,SEEK_SET)==0);
    assert(fread(&ehdr,1,sizeof(Elf64_Ehdr),fd)==sizeof(Elf64_Ehdr));
    assert(*(uint32_t *)ehdr.e_ident==0x464c457f);
    
    // search strtab
    for(int i=ehdr.e_shnum-1;i>=0;i--){
        assert(fseek(fd,ehdr.e_shoff+i*sizeof(Elf64_Shdr),SEEK_SET)==0);
        assert(fread(&shdr,1,sizeof(Elf64_Shdr),fd)==sizeof(Elf64_Shdr));
        if(shdr.sh_type==SHT_STRTAB){
            strtab_shdr = shdr;
            assert(fseek(fd,strtab_shdr.sh_offset,SEEK_SET)==0);
            strtab = (char *)malloc(strtab_shdr.sh_size);
            assert(fread(strtab,1,strtab_shdr.sh_size,fd)==strtab_shdr.sh_size);
            break;
        }
    }
    assert(strtab && "fail to extract strtab\n");
    for(int i=0;i<ehdr.e_shnum;i++){
        assert(fseek(fd,ehdr.e_shoff+i*sizeof(Elf64_Shdr),SEEK_SET)==0);
        assert(fread(&shdr,sizeof(Elf64_Shdr),1,fd)==1);
        if(strcmp(&strtab[shdr.sh_name],sec_name)==0){
            //printf("%s\n",&strtab[shdr.sh_name]);
            assert(fseek(fd,shdr.sh_offset,SEEK_SET)==0);
            *nv_fatbin = (uint8_t*)malloc(shdr.sh_size);
            *size = shdr.sh_size;
            assert(fread(*nv_fatbin,1,shdr.sh_size,fd)==shdr.sh_size);
            break;
        }
    }
    assert(nv_fatbin && "fail to extract nv_fatbin\n");
}

int main(){
    char name[100] = "bin/dummy_inv";
    char sec_name[100] = ".nv_fatbin";
    uint8_t *nv_fatbin = NULL;
    unsigned size;
    printf("reading %s\n",name);
    readelf(name,sec_name,&nv_fatbin,&size);
    printf("extract %s to pointer:%p size:%d\n",sec_name,nv_fatbin,size);
    
    for(int i=0;i<size;i+=8){
        printf("%016lx\n",*(uint64_t *)(&nv_fatbin[i]));
    }
    
    char outname[100];outname[0] = 0;strcpy(outname,name);strcat(outname,"_fatbin");
    printf("writing to %s\n",outname);
    FILE *fd = fopen(outname,"w");
    assert(fwrite(nv_fatbin,1,size,fd)==size);
    fclose(fd);
}