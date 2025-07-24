use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use sprs::{TriMat};
use rayon::prelude::*;
use rayon::{ThreadPoolBuilder, ThreadPool};
use pyo3::prelude::*;
use pyo3::ffi::c_str;

#[pyclass]
pub struct RsOneHotEncoder {
    encoder: OneHotEncoder
}

#[pymethods]
impl RsOneHotEncoder {
    #[new]
    pub fn new(data: Vec<String>, n_jobs: usize) -> PyResult<Self> {
        let encoder = OneHotEncoder::new(data, n_jobs);
        Ok(RsOneHotEncoder {encoder})
    }

    #[staticmethod]
    pub fn from_map(map: HashMap<String, usize>, n_jobs: usize) -> PyResult<Self> {
        let encoder = OneHotEncoder::from_map(map, n_jobs);
        Ok(RsOneHotEncoder {encoder})
    }

    pub fn to_single_thread(&mut self) {
        self.encoder.to_single_thread();
    }

    pub fn get_map(&self) -> PyResult<HashMap<String, usize>> {
        Ok(self.encoder.get_map())
    }

    pub fn encode(&self, data: Vec<String>) -> PyResult<PyObject> {
        let res = self.encoder.encode(data);
        let rows = res.row_inds();
        let colls = res.col_inds();
        let data = res.data();
        let shape = res.shape();

        Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code(py,
               c_str!("
from scipy.sparse import csr_matrix

def _rs_one_hot_convert_to_csr(shape, rows, colls, data):
    return csr_matrix((data, (rows, colls)), shape=shape)"),
            c_str!(""),
         c_str!("")
            ).unwrap().getattr("_rs_one_hot_convert_to_csr").unwrap().into();
            let res = fun.call1(py, (shape, rows, colls, data)).unwrap();
            Ok(res)
        })
    }
}

#[pymodule]
mod rs_fast_one_hot {
    #[pymodule_export]
    use super::RsOneHotEncoder;
}

#[derive(Debug)]
pub struct OneHotEncoder {
    label2id: Arc<RwLock<HashMap<String, usize>>>,
    executor: Arc<Mutex<Option<ThreadPool>>>
}

impl OneHotEncoder {

    pub fn from_map(map: HashMap<String, usize>, n_jobs: usize) -> OneHotEncoder {
        let label2id = Arc::new(RwLock::new(map));
        let executor = Self::create_executor(n_jobs);
        OneHotEncoder { label2id, executor }
    }

    pub fn new(data: Vec<String>, n_jobs: usize) -> OneHotEncoder {
        let label2id = Arc::new(RwLock::new(HashMap::new()));
        let executor= Self::create_executor(n_jobs);
        match executor.clone().lock().unwrap().as_ref() {
            None => {
                data.into_iter().for_each(|v| {
                    if !label2id.read().unwrap().contains_key(&v) {
                        let mut lm = label2id.write().unwrap();
                        let sz = lm.len();
                        lm.insert(v.clone(), sz);
                    }
                });
            },
            Some(exec) => {
                exec.install(|| {
                    data.into_par_iter().for_each(|v| {
                        if !label2id.read().unwrap().contains_key(&v) {
                            let mut lm = label2id.write().unwrap();
                            let sz = lm.len();
                            lm.insert(v.clone(), sz);
                        }
                    });
                })
            }
        }
        OneHotEncoder { label2id, executor }
    }

    fn create_executor(n_jobs: usize) -> Arc<Mutex<Option<ThreadPool>>> {
        let executor;
        if n_jobs > 1 {
            executor = Arc::new(Mutex::new(Some(ThreadPoolBuilder::new().num_threads(n_jobs).build().unwrap())));
        } else {
            executor = Arc::new(Mutex::new(None));
        }
        executor
    }

    pub fn to_single_thread(&mut self) {
        let mut e = self.executor.lock().unwrap();
        *e = None;
    }
    
    pub fn get_map(&self) -> HashMap<String, usize> {
        self.label2id.read().unwrap().clone()
    }



    pub fn encode(&self, data: Vec<String>) -> TriMat<i8> {
        let shape = (data.len(), self.label2id.read().unwrap().len());
        match self.executor.clone().lock().unwrap().as_ref() {
            None => {
                let m = self.label2id.read().unwrap();
                let (rows, colls): (Vec<usize>, Vec<usize>) = data.into_iter().enumerate()
                    .filter(|(_, v)| m.contains_key(v))
                    .map(|(i, v)| (i, *m.get(&v).unwrap())).unzip();
                let mut d = Vec::with_capacity(rows.len());
                d.resize(rows.len(), 1);
                TriMat::from_triplets(shape, rows, colls, d)
            },
            Some(exec) => {
                exec.install(|| {
                    let m = self.label2id.read().unwrap();
                    let (rows, colls): (Vec<usize>, Vec<usize>) = data.into_par_iter().enumerate()
                        .filter(|(_, v)| m.contains_key(v))
                        .map(|(i, v)| (i, *m.get(&v).unwrap())).unzip();
                    let mut d = Vec::with_capacity(rows.len());
                    d.resize(rows.len(), 1);
                    TriMat::from_triplets(shape, rows, colls, d)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let ohe = OneHotEncoder::new(vec!["1d".to_string(), "2s".to_string(), "2s".to_string(), "10k".to_string()], 2);
        println!("{:?}", ohe);
        let r = ohe.encode(vec![
            "1d".to_string(),
            "1d".to_string(),
            "1d".to_string(),
            "10k".to_string(),
            "1000".to_string(),
            "10k".to_string()]);
        println!("{:?}", r.col_inds());
        println!("{:?}", r.row_inds());
        println!("{:?}", r.data());
        println!("{:?}", r.shape());
    }
}
