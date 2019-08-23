#[derive(Clone)]
pub enum ComputeEvent {
    Start,
    End,
    Progress((u32, u32)),
}
