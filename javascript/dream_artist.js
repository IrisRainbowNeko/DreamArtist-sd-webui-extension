const DA_PROGRESSBAR_LABEL = 'da_preview'
const DA_GALLERY_LABEL = 'da_gallery'
const DA_ERROR_LABEL = '#da_error'
const DA_GALLERY_CHILD = 'da_gallery_kid';
const DA_PROGRESS_LABEL = 'da_progress';

function start_training_dreamartist() {
    rememberGallerySelection(DA_GALLERY_LABEL)
    gradioApp().querySelector('#da_error').innerHTML = ''
    var daGalleryElt = gradioApp().getElementById(DA_GALLERY_LABEL)
    // set id of first child of daGalleryElt to 'da_gallery_kid',
    // required by AUTOMATIC1111 UI Logic
    daGalleryElt.children[0].id = DA_GALLERY_CHILD
    var id = randomId();
    requestProgress(id,
        gradioApp().getElementById(DA_GALLERY_LABEL),
        gradioApp().getElementById(DA_GALLERY_CHILD),
        function () {
        },
        function (progress) {
            gradioApp().getElementById(DA_PROGRESS_LABEL).innerHTML = progress.textinfo
        })

    const argsToArray = args_to_array(arguments);
    argsToArray.push(argsToArray[0])
    argsToArray[0] = id;
    return argsToArray
}

