import folder_paths
from .modules.fooocus_upscale import perform_upscale
from huggingface_hub import hf_hub_download

local_dir = "upscale_models"
repo_id = "lllyasviel/misc"
file_name = "fooocus_upscaler_s409985e5.bin"
class FooocusUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
        }

    #  If the node is output node, set this to True.
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self,
               image):
        
        model_path = folder_paths.get_full_path(local_dir, file_name)
        if model_path==None:
            print(f"{file_name} not found in {folder_paths.get_folder_paths(local_dir)[0]},downloading...")
            hf_hub_download(repo_id,file_name,local_dir=folder_paths.get_folder_paths(local_dir)[0])
            model_path = folder_paths.get_full_path(local_dir, file_name)
            
        return (perform_upscale(image,model_path), )
